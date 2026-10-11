"""토큰화부터 BERT 본체와 분류층 학습, 검증, 저장까지 차례로 실행한다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step04_train_classifier.py
uv run python steps/step04_train_classifier.py --lr 1e-5 2e-5 5e-5      # 학습률 비교
uv run python steps/step04_train_classifier.py --model bert roberta electra-s --lr 1e-5 2e-5 5e-5
uv run python steps/step04_train_classifier.py --model roberta --lr 2e-5 --seed 42 43 44
uv run python steps/step04_train_classifier.py --top-k 3 --extra-seeds 43 44   # 상위 설정 반복

모델 이름은 lesson_settings.CLASSIFIER_CANDIDATES의 짧은 이름이나 Hugging Face ID를 쓴다.
여러 조합을 주면 모델 × 학습률 × seed 순서로 차례로 학습하고, 하나가 실패해도 계속한다.
--top-k는 팀 전체 결과에서 validation 평균이 높은 설정 K개를 골라, 아직 없는 seed만 학습한다.
"""

import argparse
import gc
import json
import math
import random
import time
import traceback
from itertools import product
from pathlib import Path

import numpy as np
import torch
from common.metrics import classification_scores, save_errors
from common.plots import plot_learning_curve
from common.records import new_run_dir, now_iso, write_json
from common.resources import (
    environment_info,
    gpu_peak_mb,
    measure_latency,
    process_rss_mb,
    reset_gpu_peak,
)
from lesson_settings import (
    CLASSIFIER_CANDIDATES,
    CLASSIFIER_MODEL,
    DATASET,
    LEARNER_DIR,
    OUTPUT_ROOT,
    ROOT,
)
from step01_read_data import read_labels, read_rows
from step02_check_data import check_splits
from step05_select_model import collect_candidates, group_candidates
from step06_predict import load_run, predict_text
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

# 학습률 비교에서는 학습률만 바꾸고 다른 조건은 고정한다.
DATA_DIR = ROOT / "data" / DATASET
LEARNING_RATE = 2e-5
EPOCHS = 5
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
MAX_LENGTH = 192
SEED = 42
DEVICE = "cuda"  # GPU를 사용한다. CPU 실행이 필요하면 명시적으로 cpu로 바꾼다.


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, labels, max_length):
        # 아직 padding하지 않는다. DataLoader가 배치로 묶을 때 길이를 맞춘다.
        self.features = tokenizer([row["text"] for row in rows],
                                  truncation=True, max_length=max_length, padding=False)
        label2id = {label: index for index, label in enumerate(labels)}
        self.targets = [label2id[row["label"]] for row in rows]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # input_ids와 attention_mask, 그리고 정답 번호 하나를 반환한다.
        return {**{key: value[index] for key, value in self.features.items()},
                "labels": self.targets[index]}


def preview_training_batch(rows, tokenizer, dataset, collator):
    """앞의 두 행을 직접 묶어 입력을 관찰한다. 학습용 loader는 순회하지 않는다."""
    count = min(2, len(dataset))
    original_lengths = []
    truncated_lengths = []
    for index in range(count):
        original = tokenizer(rows[index]["text"], truncation=False)
        original_lengths.append(len(original["input_ids"]))
        truncated_lengths.append(len(dataset[index]["input_ids"]))
        print(f"입력 {index + 1} 원문:", rows[index]["text"])
        print("특수 토큰 포함 길이:", original_lengths[-1],
              "길이 제한 적용 후:", truncated_lengths[-1])
    # DataLoader의 shuffle 순서나 난수 상태를 소비하지 않고 같은 collator만 사용한다.
    batch = collator([dataset[index] for index in range(count)])
    shapes = {key: list(value.shape) for key, value in batch.items()}
    print("두 행을 묶은 Tensor 크기:", shapes)
    print("정답 번호:", batch["labels"].tolist())
    return {"original_lengths": original_lengths, "truncated_lengths": truncated_lengths,
            "shapes": shapes, "labels": batch["labels"].tolist()}


def token_length_stats(rows, tokenizer, max_length):
    # 입력 길이 제한 때문에 잘리는 문장이 얼마나 되는지 기록한다.
    lengths = [len(ids) for ids in tokenizer([row["text"] for row in rows],
                                             truncation=False)["input_ids"]]
    return {"max": max(lengths), "mean": sum(lengths) / len(lengths),
            "truncated_rows": sum(length > max_length for length in lengths)}


def validate_model(model, loader, device, labels):
    model.eval()
    predictions, actual = [], []
    total_loss, count = 0.0, 0
    # 검증에서는 정답으로 점수를 계산하지만 가중치를 업데이트하지 않는다.
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            batch_size = batch["labels"].shape[0]
            total_loss += output.loss.item() * batch_size
            count += batch_size
            predictions.extend(output.logits.argmax(-1).cpu().tolist())
            actual.extend(batch["labels"].cpu().tolist())
    return classification_scores(actual, predictions, labels), total_loss / count, predictions


def resolve_model(name):
    """짧은 이름이면 Hugging Face ID로 바꾼다. 실행 이름에는 짧은 이름을 쓴다."""
    if name in CLASSIFIER_CANDIDATES:
        return name, CLASSIFIER_CANDIDATES[name]
    for alias, model_id in CLASSIFIER_CANDIDATES.items():
        if model_id == name:
            return alias, model_id
    return str(name).split("/")[-1], str(name)


def default_run_name(alias, learning_rate, seed):
    return f"{alias}_lr{learning_rate:.0e}_s{seed}".replace("e-0", "e-")


def run_training(data_dir=DATA_DIR, learner_dir=LEARNER_DIR, run_name=None,
                 model_name=CLASSIFIER_MODEL, learning_rate=LEARNING_RATE, epochs=EPOCHS,
                 batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS,
                 max_length=MAX_LENGTH, seed=SEED, device_name=DEVICE,
                 gradient_checkpointing=True, local_files_only=False):
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    if min(epochs, batch_size, accumulation_steps, max_length) < 1 or not 0 < learning_rate < 1:
        raise ValueError("학습 횟수와 크기는 양수, 학습률은 0과 1 사이여야 합니다.")
    model_alias, model_name = resolve_model(model_name)
    run_name = run_name or default_run_name(model_alias, learning_rate, seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 환경 또는 DEVICE 설정을 확인하세요.")

    # 1. CSV와 라벨을 읽고 train과 validation을 확인한다. test는 읽지 않는다.
    labels = read_labels(data_dir)
    train_rows = read_rows(data_dir / "train.csv")
    validation_rows = read_rows(data_dir / "validation.csv")
    check_splits({"train": train_rows, "validation": validation_rows}, labels)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 모델 가중치, optimizer 상태, 활성값을 모두 포함한 최대 VRAM을 재기 위해 먼저 초기화한다.
    reset_gpu_peak()

    # 2. 토크나이저와 분류 모델을 같은 이름에서 불러온다.
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
        local_files_only=local_files_only)
    # 본체를 고정하지 않는다. optimizer에는 본체와 분류층의 파라미터가 모두 들어간다.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    use_checkpointing = gradient_checkpointing and device.type == "cuda"
    if use_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)

    # 3. 배치마다 가장 긴 입력에 맞춰 padding한다. 길이는 8의 배수로 맞춘다.
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    train_dataset = TextDataset(train_rows, tokenizer, labels, max_length)
    preview_training_batch(train_rows, tokenizer, train_dataset, collator)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True, collate_fn=collator,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    validation_loader = DataLoader(TextDataset(validation_rows, tokenizer, labels, max_length),
                                   batch_size=batch_size, shuffle=False,
                                   collate_fn=collator, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_updates = epochs * math.ceil(len(train_loader) / accumulation_steps)
    warmup_updates = max(1, int(total_updates * 0.1))

    def learning_rate_factor(step):
        if step < warmup_updates:
            return (step + 1) / warmup_updates
        return max(0.0, (total_updates - step) / max(1, total_updates - warmup_updates))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    use_cuda = device.type == "cuda"
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    precision = ("BF16 autocast" if dtype == torch.bfloat16 else "FP16 autocast + GradScaler"
                 ) if use_cuda else "FP32"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and dtype == torch.float16)

    output_dir = new_run_dir(learner_dir, run_name)
    config = {"kind": "transformer", "dataset": data_dir.name, "learner": learner_dir.name,
              "run_name": run_name, "model": model_name, "model_alias": model_alias,
              "learning_rate": learning_rate,
              "epochs": epochs, "batch_size": batch_size,
              "accumulation_steps": accumulation_steps, "max_length": max_length,
              "seed": seed, "device": str(device), "precision": precision,
              "gradient_checkpointing": use_checkpointing, "labels": labels}
    write_json(output_dir / "config.json", config)

    # 4. train에서 loss를 계산하고, 누적 구간 끝에서 가중치를 갱신한다.
    best_score, best_record, updates = -1.0, None, 0
    train_seconds = 0.0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_train_loss, count = 0.0, 0
        epoch_started = time.perf_counter()
        for index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            window_start = (index // accumulation_steps) * accumulation_steps
            window_size = min(accumulation_steps, len(train_loader) - window_start)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_cuda):
                output = model(**batch)
                # 마지막 누적 구간은 배치 수가 적을 수 있다.
                loss = output.loss / window_size
            scaler.scale(loss).backward()
            batch_count = batch["labels"].shape[0]
            total_train_loss += output.loss.item() * batch_count
            count += batch_count
            if (index + 1) % accumulation_steps == 0 or index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)  # 가중치가 실제로 바뀌는 지점
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        if use_cuda:
            torch.cuda.synchronize()
        train_seconds += time.perf_counter() - epoch_started

        # 5. validation으로 현재 모델을 평가한다. 검증 함수에는 backward가 없다.
        validation_started = time.perf_counter()
        metrics, validation_loss, predictions = validate_model(
            model, validation_loader, device, labels)
        validation_seconds = time.perf_counter() - validation_started
        record = {"epoch": epoch, "train_loss": total_train_loss / count,
                  "validation_loss": validation_loss,
                  "validation_macro_f1": metrics["macro_f1"], "optimizer_steps": updates,
                  "validation_seconds": validation_seconds,
                  "elapsed_seconds": time.perf_counter() - started}
        with (output_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        # 6. 더 높은 validation 점수를 얻었을 때만 가중치를 저장한다.
        if metrics["macro_f1"] > best_score:
            best_score, best_record = metrics["macro_f1"], record
            checkpoint = output_dir / "checkpoint"
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            summary = {"method": "full_backbone_and_head_finetuning", "best_epoch": epoch,
                       "optimizer_steps_at_checkpoint": updates,
                       "selection_split": "validation", "max_length": max_length}
            write_json(output_dir / "training_summary.json", summary)
            write_json(output_dir / "validation_metrics.json", metrics)
            save_errors(output_dir / "validation_errors.csv", validation_rows, predictions, labels)

    total_seconds = time.perf_counter() - started
    peaks = gpu_peak_mb()
    plot_learning_curve(output_dir / "epochs.jsonl", output_dir / "learning_curve.png",
                        f"{model_name} lr={learning_rate} ({output_dir.name})")

    # 7. GPU 메모리를 비운 뒤, 서비스와 같은 조건(CPU, 1건씩)으로 추론 시간을 잰다.
    del model, optimizer, scheduler
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    bundle = load_run(output_dir)
    latency = measure_latency(lambda text: predict_text(text, bundle),
                              [row["text"] for row in validation_rows])
    best_metrics = json.loads((output_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    experiment = {
        "experiment_id": output_dir.name, "learner": learner_dir.name,
        "dataset": data_dir.name, "kind": "transformer",
        "model_info": {
            "model_id": model_name, "model_alias": model_alias,
            "model_size_params": parameter_count,
            "precision": precision, "training_type": "Full Fine-tuning (본체 + 분류층)",
            "gradient_checkpointing": use_checkpointing,
            "pretrained_source": f"https://huggingface.co/{model_name}",
            "tokenizer": type(tokenizer).__name__,
        },
        "hyperparameters": {
            "max_length": max_length, "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
            "learning_rate": learning_rate, "epochs": epochs, "seed": seed,
            "optimizer": "AdamW(weight_decay=0.01)",
            "scheduler": "linear warmup 10% + linear decay", "grad_clip": 1.0,
        },
        "token_lengths": {"train": token_length_stats(train_rows, tokenizer, max_length),
                          "validation": token_length_stats(validation_rows, tokenizer,
                                                           max_length)},
        "resource_and_time": {
            "train_time_sec": train_seconds,
            "total_time_sec_including_validation": total_seconds,
            "inference_latency_per_sample_ms": latency["mean_ms"],
            "inference_latency": {**latency, "device": "cpu"},
            "gpu_validation_ms_per_sample_at_best_epoch":
                best_record["validation_seconds"] / len(validation_rows) * 1000,
            "peak_vram_mb": peaks["peak_vram_allocated_mb"],
            "peak_vram_reserved_mb": peaks["peak_vram_reserved_mb"],
            "process_rss_mb": process_rss_mb(),
        },
        "evaluation_metrics": {
            "split": "validation", "best_epoch": best_record["epoch"],
            "accuracy": best_metrics["accuracy"], "macro_f1": best_metrics["macro_f1"],
            "weighted_f1": best_metrics["weighted_f1"],
            "confusion_matrix": best_metrics["confusion_matrix"],
        },
        "environment": environment_info(),
        "timestamp": now_iso(),
    }
    write_json(output_dir / "experiment.json", experiment)
    print(f"최고 validation macro F1 {best_score:.4f} (epoch {best_record['epoch']}), "
          f"학습 {train_seconds:.1f}초, peak VRAM {peaks['peak_vram_allocated_mb'] or 0:.0f}MB, "
          f"CPU 1건 추론 {latency['mean_ms']:.1f}ms")
    print("저장 위치:", output_dir)
    return output_dir


def top_config_jobs(top_k, extra_seeds, dataset_dir=OUTPUT_ROOT / DATASET):
    """validation 평균 상위 설정을 골라, 그 설정에 아직 없는 seed의 학습 목록을 만든다."""
    candidates = [row for row in collect_candidates(dataset_dir) if row["kind"] == "transformer"]
    if not candidates:
        raise RuntimeError("반복할 사전학습 모델 실행이 없습니다. 먼저 --model, --lr로 학습하세요.")
    groups = sorted(group_candidates(candidates), key=lambda group: -group["macro_f1_mean"])
    jobs = []
    for group in groups[:top_k]:
        row = next(item for item in candidates if item["run_dir"] == group["runs"][0])
        missing = [seed for seed in extra_seeds if seed not in group["seeds"]]
        print(f"상위 설정: {group['model_alias']} lr={row['learning_rate']} "
              f"평균 macro F1 {group['macro_f1_mean']:.4f}, 기존 seed {group['seeds']}, "
              f"추가 seed {missing}")
        # seed 외의 설정은 원래 실행과 똑같이 맞춘다.
        jobs += [{"model_name": row["model"], "learning_rate": row["learning_rate"],
                  "epochs": row["epochs"], "batch_size": row["batch_size"],
                  "accumulation_steps": row["accumulation_steps"],
                  "max_length": row["max_length"], "seed": seed} for seed in missing]
    return jobs


def main():
    parser = argparse.ArgumentParser(description="사전학습 분류 모델 파인튜닝")
    parser.add_argument("--model", nargs="+", default=[CLASSIFIER_MODEL],
                        help=f"짧은 이름({', '.join(CLASSIFIER_CANDIDATES)}) 또는 Hugging Face ID")
    parser.add_argument("--lr", type=float, nargs="+", default=[LEARNING_RATE])
    parser.add_argument("--seed", type=int, nargs="+", default=[SEED])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accumulation", type=int, default=ACCUMULATION_STEPS)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True, help="끄면 VRAM은 늘고 속도는 빨라진다.")
    parser.add_argument("--run-name", help="조합이 하나일 때만 쓸 실행 이름")
    parser.add_argument("--top-k", type=int,
                        help="validation 평균 상위 K개 설정을 --extra-seeds로 다시 학습한다.")
    parser.add_argument("--extra-seeds", type=int, nargs="+", default=[43, 44])
    args = parser.parse_args()

    if args.top_k:
        # 상위 설정 반복에서는 --model, --lr, --seed와 학습 설정을 기존 실행에서 가져온다.
        jobs = top_config_jobs(args.top_k, args.extra_seeds)
    else:
        jobs = [{"model_name": model_name, "learning_rate": learning_rate, "seed": seed,
                 "epochs": args.epochs, "batch_size": args.batch_size,
                 "accumulation_steps": args.accumulation, "max_length": args.max_length}
                for model_name, learning_rate, seed in product(args.model, args.lr, args.seed)]
    if args.run_name and len(jobs) > 1:
        parser.error("--run-name은 모델, 학습률, seed 조합이 하나일 때만 쓸 수 있습니다.")

    finished, failed = [], []
    for index, job in enumerate(jobs, start=1):
        label = f"{job['model_name']}, lr={job['learning_rate']}, seed={job['seed']}"
        print(f"\n===== {index}/{len(jobs)}: {label} =====")
        try:
            finished.append(run_training(run_name=args.run_name, device_name=args.device,
                                         gradient_checkpointing=args.gradient_checkpointing,
                                         **job))
        except Exception:  # noqa: BLE001 - 한 조합이 실패해도 나머지 실험은 계속한다.
            # 실패한 실행 폴더에는 experiment.json이 없으므로 step05 후보에서 빠진다.
            traceback.print_exc()
            failed.append(label)
        # 다음 실험이 앞 실험의 GPU 메모리에 영향받지 않게 정리한다.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n완료 {len(finished)}개, 실패 {len(failed)}개")
    for label in failed:
        print(f"  실패: {label}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
