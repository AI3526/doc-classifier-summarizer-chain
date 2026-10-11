"""검수용: 데이터 원본과 실제 가중치 변경, 선정 규칙 준수를 수업 코드 밖에서 확인한다."""

import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

from lesson_settings import DATA_DIR, OUTPUT_ROOT, SELECTION_DIR
from step05_select_model import choose


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def data_snapshot(data_dir):
    """학습 전후에 호출해 같은 train, validation, 라벨을 썼는지 비교한다."""
    return {name: hashlib.sha256((Path(data_dir) / name).read_bytes()).hexdigest()
            for name in ("train.csv", "validation.csv", "labels.json")}


def check_original_data(data_dir):
    """배포 manifest와 현재 파일을 대조한다. test 입력은 읽지 않는다.

    data/manifest.json은 이전 데이터 버전 기준이라 현재 v2 데이터와 다를 수 있다.
    그래서 실패로 멈추지 않고 일치 여부를 기록한다.
    """
    data_dir = Path(data_dir)
    snapshot = data_snapshot(data_dir)
    topics = read_json(data_dir.parent / "manifest.json").get("topics", {})
    expected = topics.get(data_dir.name, {}).get("files", {})
    return {name: {"sha256": digest,
                   "manifest_match": expected.get(name, {}).get("sha256") == digest}
            for name, digest in snapshot.items()}


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
    return {"run_dir": run_dir.relative_to(OUTPUT_ROOT).as_posix(), "changed": changed,
            "initial": initial, "selected_checkpoint": current, "best_epoch": best["epoch"],
            "validation_macro_f1": metrics["macro_f1"]}


def verify_selection(selection_dir):
    """선정 결과가 기록된 규칙과 실험 점수를 그대로 따랐는지 확인한다."""
    selection = read_json(Path(selection_dir) / "selected.json")
    candidates = selection["candidates"]
    if selection["selected_by"] != "validation":
        raise ValueError("validation으로 선택한 기록이 필요합니다.")
    if {row["kind"] for row in candidates} < {"baseline", "transformer"}:
        raise ValueError("기준 모델과 사전학습 모델이 모두 후보에 있어야 합니다.")
    for candidate in candidates:
        metrics = read_json(OUTPUT_ROOT / candidate["run_dir"] / "validation_metrics.json")
        if candidate["validation_macro_f1"] != metrics["macro_f1"]:
            raise ValueError(f"선택 당시 기록과 실험 점수가 다릅니다: {candidate['run_dir']}")
    if len({tuple(row["labels"]) for row in candidates}) != 1:
        raise ValueError("같은 라벨로 학습한 실험끼리 비교하세요.")
    if choose(candidates)["run_dir"] != selection["selected"]["run_dir"]:
        raise ValueError("기록된 선정 규칙으로 다시 골랐을 때 결과가 다릅니다.")
    return {"selected_run": selection["selected"]["run_dir"], "candidates": len(candidates)}


def main():
    selection = read_json(SELECTION_DIR / "selected.json")
    report = {"data": check_original_data(DATA_DIR),
              "selection": verify_selection(SELECTION_DIR), "runs": []}
    for candidate in selection["candidates"]:
        if candidate["kind"] == "transformer" and candidate["weights_available"]:
            report["runs"].append(verify_run(OUTPUT_ROOT / candidate["run_dir"]))
    target = SELECTION_DIR / "learning_verification.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("데이터 원본, 학습 가중치, 선택 기록 검수 완료:", target)


if __name__ == "__main__":
    main()
