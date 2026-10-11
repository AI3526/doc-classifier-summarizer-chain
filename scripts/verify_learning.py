"""튜터 검수용: 데이터 원본과 실제 가중치 변경을 수업 코드 밖에서 확인한다."""

import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

from lesson_settings import DATA_DIR, LEARNER_DIR


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def data_snapshot(data_dir):
    """학습 전후에 호출해 같은 train, validation, 라벨을 썼는지 비교한다."""
    return {name: hashlib.sha256((Path(data_dir) / name).read_bytes()).hexdigest()
            for name in ("train.csv", "validation.csv", "labels.json")}


def verify_original_data(data_dir):
    """배포 manifest와 현재 파일을 대조한다. test 입력은 읽지 않는다."""
    data_dir = Path(data_dir)
    expected = read_json(data_dir.parent / "manifest.json")["topics"][data_dir.name]["files"]
    snapshot = data_snapshot(data_dir)
    for name, digest in snapshot.items():
        if digest != expected[name]["sha256"]:
            raise ValueError(f"배포 원본과 데이터가 다릅니다: {data_dir.name}/{name}")
    return snapshot


def parameter_evidence(model):
    """가중치를 CPU에서 읽어 본체와 분류층의 해시를 따로 계산한다."""
    backbone_ids = {id(parameter) for parameter in model.base_model.parameters()}
    digests = {"backbone": hashlib.sha256(), "head": hashlib.sha256()}
    counts = {"backbone": 0, "head": 0}
    for name, parameter in model.named_parameters():
        part = "backbone" if id(parameter) in backbone_ids else "head"
        digests[part].update(name.encode())
        digests[part].update(parameter.detach().float().cpu().numpy().tobytes())
        counts[part] += parameter.numel()
    return {part: {"sha256": digest.hexdigest(), "parameters": counts[part]}
            for part, digest in digests.items()}


def verify_run(run_dir):
    """같은 초기 모델과 저장한 모델을 비교한다. 모델 다운로드나 학습은 하지 않는다."""
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "training_summary.json")
    labels = config["labels"]
    # 학습 때와 같은 시드로 새 분류층을 초기화한다. 호출자의 CPU 난수 상태는 보존한다.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        original = AutoModelForSequenceClassification.from_pretrained(
            config["model"], num_labels=len(labels), id2label=dict(enumerate(labels)),
            label2id={label: index for index, label in enumerate(labels)}, local_files_only=True)
    initial = parameter_evidence(original)
    del original
    saved = AutoModelForSequenceClassification.from_pretrained(
        run_dir / "checkpoint", local_files_only=True, use_safetensors=True)
    actual_labels = [saved.config.id2label[index] for index in range(saved.config.num_labels)]
    if actual_labels != labels or saved.config.label2id != dict(zip(labels, range(len(labels)))):
        raise ValueError("학습 설정과 저장 모델의 라벨 순서가 다릅니다.")
    current = parameter_evidence(saved)
    del saved
    changed = {part: initial[part]["sha256"] != current[part]["sha256"] for part in initial}
    if not all(changed.values()):
        raise ValueError("본체와 분류층이 모두 학습되었는지 확인하세요.")
    epochs = [json.loads(line) for line in
              (run_dir / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    best = max(epochs, key=lambda row: row["validation_macro_f1"])
    metrics = read_json(run_dir / "validation_metrics.json")
    if (summary["selection_split"] != "validation" or summary["best_epoch"] != best["epoch"]
            or summary["optimizer_steps_at_checkpoint"] != best["optimizer_steps"]
            or summary["optimizer_steps_at_checkpoint"] < 1
            or metrics["macro_f1"] != best["validation_macro_f1"]):
        raise ValueError("저장한 epoch와 validation 기록이 일치하지 않습니다.")
    if summary["max_length"] != config["max_length"]:
        raise ValueError("학습 설정과 요약의 입력 길이가 다릅니다.")
    return {"run_name": run_dir.name, "changed": changed, "initial": initial,
            "selected_checkpoint": current, "best_epoch": best["epoch"],
            "validation_macro_f1": metrics["macro_f1"]}


def verify_selection(learner_dir):
    """선택한 실험이 validation 최고점인지와 실험 설정 차이를 확인한다."""
    learner_dir = Path(learner_dir)
    selection = read_json(learner_dir / "selected.json")
    candidates = selection["candidates"]
    if selection["selected_by"] != "validation_macro_f1":
        raise ValueError("validation으로 선택한 기록이 필요합니다.")
    settings = set()
    labels = set()
    datasets = set()
    for candidate in candidates:
        run = learner_dir.parent / candidate["run_learner"] / candidate["run_name"]
        config = read_json(run / "config.json")
        metrics = read_json(run / "validation_metrics.json")
        if candidate["validation_macro_f1"] != metrics["macro_f1"]:
            raise ValueError("선택 당시 기록과 실험 점수가 다릅니다.")
        settings.add(tuple(config[key] for key in (
            "model", "learning_rate", "epochs", "batch_size", "accumulation_steps", "max_length")))
        labels.add(tuple(config["labels"]))
        datasets.add(config["dataset"])
    if len(settings) < 2:
        raise ValueError("이름 외에 실제 학습 설정도 달라야 합니다.")
    if len(labels) != 1 or len(datasets) != 1:
        raise ValueError("같은 데이터와 라벨로 학습한 실험끼리 비교하세요.")
    best = max(candidates, key=lambda row: row["validation_macro_f1"])
    if any(selection[key] != best[key] for key in ("run_name", "run_learner")):
        raise ValueError("validation 최고점과 선택한 실험이 다릅니다.")
    return {"selected_run": best["run_name"], "different_configurations": len(settings)}


def main():
    selection = read_json(LEARNER_DIR / "selected.json")
    report = {"data_sha256": verify_original_data(DATA_DIR),
              "selection": verify_selection(LEARNER_DIR), "runs": []}
    for candidate in selection["candidates"]:
        run = LEARNER_DIR.parent / candidate["run_learner"] / candidate["run_name"]
        report["runs"].append(verify_run(run))
    target = LEARNER_DIR / "learning_verification.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("데이터 원본, 학습 가중치, 선택 기록 검수 완료:", target)


if __name__ == "__main__":
    main()
