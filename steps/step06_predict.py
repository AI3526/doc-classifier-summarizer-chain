"""선택한 분류기를 불러와 새 문장을 분류한다.

기준 모델(TF-IDF + 로지스틱 회귀)과 사전학습 모델(BERT 등) 중 무엇이 선택되어도
같은 함수로 불러오고 같은 형식으로 예측한다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step06_predict.py
"""

import json
from pathlib import Path

import joblib
import torch
from common.records import read_json
from lesson_settings import MAX_INPUT_CHARS, OUTPUT_ROOT, SELECTION_DIR
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SAMPLE_TEXT = "다음 주 월요일 오전 10시에 시스템 점검을 진행합니다. 점검 중에는 로그인이 제한됩니다."


def load_run(run_dir) -> dict:
    """실행 폴더 하나를 CPU에서 불러온다. 사전학습 모델을 새로 내려받지 않는다."""
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    labels = config["labels"]
    identity = {"dataset": config["dataset"], "learner": config["learner"],
                "run_name": run_dir.name, "kind": config["kind"], "device": "cpu"}
    bundle = {"kind": config["kind"], "labels": labels, "identity": identity,
              "run_dir": run_dir, "max_length": config.get("max_length")}

    if config["kind"] == "baseline":
        model_path = run_dir / "baseline.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"저장된 기준 모델이 없습니다: {model_path}")
        return {**bundle, "model": joblib.load(model_path)}

    checkpoint = run_dir / "checkpoint"
    if not (checkpoint / "model.safetensors").exists():
        raise FileNotFoundError(
            f"저장된 가중치가 없습니다: {checkpoint}\n"
            "다른 팀원의 실행이면 가중치를 받아 오거나 같은 설정으로 다시 학습하세요.")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, use_safetensors=True).to("cpu").eval()
    if [model.config.id2label[index] for index in range(model.config.num_labels)] != labels:
        raise ValueError("학습 기록과 체크포인트의 라벨 순서가 다릅니다.")
    return {**bundle, "tokenizer": tokenizer, "model": model}


def load_classifier(selection_dir=SELECTION_DIR) -> dict:
    selected_path = Path(selection_dir) / "selected.json"
    if not selected_path.exists():
        raise RuntimeError("먼저 step05_select_model.py에서 사용할 실험을 고르세요.")
    selected = read_json(selected_path)
    return load_run(OUTPUT_ROOT / selected["selected"]["run_dir"])


def predict_text(text: str, bundle: dict) -> dict:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"입력은 공백이 아닌 1~{MAX_INPUT_CHARS}자 문자열이어야 합니다.")
    text = text.strip()

    if bundle["kind"] == "baseline":
        # 문자 n-gram은 길이 제한이 없으므로 잘리는 정보가 없다.
        probabilities = bundle["model"].predict_proba([text])[0]
        index = int(probabilities.argmax())
        return {"label": bundle["labels"][index], "confidence": float(probabilities[index]),
                "truncated": False, "model": bundle["identity"]}

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
