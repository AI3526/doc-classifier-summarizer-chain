"""선택한 분류기의 토크나이저와 가중치를 불러와 새 문장을 분류한다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step06_predict.py
"""

import json
from pathlib import Path

import torch
from lesson_settings import LEARNER_DIR, MAX_INPUT_CHARS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SAMPLE_TEXT = "다음 주 월요일 오전 10시에 시스템 점검을 진행합니다. 점검 중에는 로그인이 제한됩니다."


def load_classifier(learner_dir=LEARNER_DIR) -> dict:
    learner_dir = Path(learner_dir)
    selected_path = learner_dir / "selected.json"
    if not selected_path.exists():
        raise RuntimeError("먼저 step05_select_model.py에서 사용할 실험을 고르세요.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    run_learner = selected.get("run_learner", learner_dir.name)
    run_name = selected["run_name"]
    run = learner_dir.parent / run_learner / run_name
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))

    # 선택한 폴더에서만 읽는다. 사전학습 모델을 새로 받아 쓰지 않는다.
    checkpoint = run / "checkpoint"
    if not (checkpoint / "model.safetensors").exists():
        raise FileNotFoundError(f"저장된 가중치가 없습니다: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, use_safetensors=True).to("cpu").eval()
    labels = config["labels"]
    if [model.config.id2label[index] for index in range(model.config.num_labels)] != labels:
        raise ValueError("학습 기록과 체크포인트의 라벨 순서가 다릅니다.")
    identity = {"dataset": config["dataset"], "learner": run_learner,
                "run_name": run_name, "device": "cpu"}
    return {"tokenizer": tokenizer, "model": model, "labels": labels,
            "max_length": config["max_length"], "identity": identity, "run_dir": run}


def predict_text(text: str, bundle: dict) -> dict:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"입력은 공백이 아닌 1~{MAX_INPUT_CHARS}자 문자열이어야 합니다.")
    text = text.strip()
    # 자르기 전 토큰 수를 확인해야 정보가 잘렸는지를 알 수 있다.
    tokenizer = bundle["tokenizer"]
    raw = tokenizer(text, truncation=False, return_attention_mask=False)
    truncated = len(raw["input_ids"]) > bundle["max_length"]
    features = tokenizer(text, truncation=True, max_length=bundle["max_length"],
                         return_tensors="pt")
    with torch.inference_mode():
        logits = bundle["model"](**features).logits  # 문장 한 개의 점수: [1, 라벨 수]
        probabilities = logits.softmax(dim=-1)[0]
    index = int(probabilities.argmax())
    return {"label": bundle["labels"][index], "confidence": float(probabilities[index]),
            "truncated": truncated, "model": bundle["identity"]}


def main():
    bundle = load_classifier()
    result = predict_text(SAMPLE_TEXT, bundle)
    print("입력:", SAMPLE_TEXT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
