"""토큰화부터 BERT 본체와 분류층 학습, 검증, 저장까지 차례로 실행한다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step04_train_classifier.py
"""

import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from lesson_settings import CLASSIFIER_MODEL, DATASET, LEARNER_DIR, ROOT
from step01_read_data import read_labels, read_rows
from step02_check_data import check_splits
from step03_train_baseline import classification_scores, save_errors
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

# 두 번째 실험은 RUN_NAME과 LEARNING_RATE를 함께 바꾼다.
# 다른 조건을 고정해야 학습률에 따른 차이를 비교할 수 있다.
DATA_DIR = ROOT / "data" / DATASET
RUN_NAME = "lr2e5"
LEARNING_RATE = 2e-5
EPOCHS = 3
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


def run_training(data_dir=DATA_DIR, learner_dir=LEARNER_DIR, run_name=RUN_NAME,
                 model_name=CLASSIFIER_MODEL, learning_rate=LEARNING_RATE, epochs=EPOCHS,
                 batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS,
                 max_length=MAX_LENGTH, seed=SEED, device_name=DEVICE,
                 local_files_only=False):
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    if min(epochs, batch_size, accumulation_steps, max_length) < 1 or not 0 < learning_rate < 1:
        raise ValueError("학습 횟수와 크기는 양수, 학습률은 0과 1 사이여야 합니다.")
    output_dir = learner_dir / run_name
    if output_dir.exists():
        raise FileExistsError(f"기존 실험을 덮어쓰지 않습니다. RUN_NAME을 바꾸세요: {output_dir}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 환경 또는 DEVICE 설정을 확인하세요.")

    # 1. CSV와 라벨을 읽고 train과 validation을 확인한다. test는 읽지 않는다.
    labels = read_labels(data_dir)
    train_rows = read_rows(data_dir / "train.csv")
    validation_rows = read_rows(data_dir / "validation.csv")
    check_splits({"train": train_rows, "validation": validation_rows}, labels)
    baseline_dir = learner_dir / "baseline"
    if not (baseline_dir / "baseline.joblib").exists():
        raise RuntimeError("먼저 step03_train_baseline.py에서 기준 모델을 학습하세요.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 2. 토크나이저와 분류 모델을 같은 이름에서 불러온다.
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
        local_files_only=local_files_only)
    # 본체를 고정하지 않는다. optimizer에는 본체와 분류층의 파라미터가 모두 들어간다.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if device.type == "cuda":
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and dtype == torch.float16)

    output_dir.mkdir(parents=True)
    config = {"dataset": data_dir.name, "learner": learner_dir.name, "run_name": run_name,
              "model": str(model_name), "learning_rate": learning_rate, "epochs": epochs,
              "batch_size": batch_size, "accumulation_steps": accumulation_steps,
              "max_length": max_length, "seed": seed, "device": str(device), "labels": labels}
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    # 최종 평가에서 같은 기준 모델을 비교할 수 있게 실험 폴더에 함께 보관한다.
    shutil.copy2(baseline_dir / "baseline.joblib", output_dir / "baseline.joblib")
    shutil.copy2(baseline_dir / "validation_metrics.json", output_dir / "baseline_validation.json")

    # 4. train에서 loss를 계산하고, 누적 구간 끝에서 가중치를 갱신한다.
    best_score, updates = -1.0, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_train_loss, count = 0.0, 0
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

        # 5. validation으로 현재 모델을 평가한다. 검증 함수에는 backward가 없다.
        metrics, validation_loss, predictions = validate_model(
            model, validation_loader, device, labels)
        record = {"epoch": epoch, "train_loss": total_train_loss / count,
                  "validation_loss": validation_loss,
                  "validation_macro_f1": metrics["macro_f1"], "optimizer_steps": updates,
                  "elapsed_seconds": time.perf_counter() - started}
        with (output_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        # 6. 더 높은 validation 점수를 얻었을 때만 가중치를 저장한다.
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            checkpoint = output_dir / "checkpoint"
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            summary = {"method": "full_backbone_and_head_finetuning", "best_epoch": epoch,
                       "optimizer_steps_at_checkpoint": updates,
                       "selection_split": "validation", "max_length": max_length}
            for filename, value in (("training_summary.json", summary),
                                    ("validation_metrics.json", metrics)):
                (output_dir / filename).write_text(
                    json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            save_errors(output_dir / "validation_errors.csv", validation_rows, predictions, labels)

    print("저장 위치:", output_dir)
    return output_dir


def main():
    run_training()


if __name__ == "__main__":
    main()
