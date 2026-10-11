"""08. 선택한 분류 모델과 정책, 실제 생성 호출을 LangChain으로 연결한다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step08_connect_chain.py
"""

import json

import lesson_settings as lesson
from langchain_core.runnables import RunnableLambda
from prompts.document_policies import POLICIES, PROMPT_VERSION, build_prompt
from step06_predict import load_classifier, predict_text
from step07_call_models import call_model

# lesson_settings.py의 학습 결과를 load_classifier가 불러온다.
INPUT_TEXT = (
    "운영 회의록: 교육장 점검 일정을 논의했다. 10월 8일에 장비를 점검하기로 결정했다. "
    "민수는 점검 목록을 작성하고, 지수는 수강생 안내문을 준비한다. 제출 기한은 미정이다."
)
PROVIDER = "ollama"
MAX_INPUT_CHARS = lesson.MAX_INPUT_CHARS


def apply_policy(item):
    # 분류 라벨을 딕셔너리의 키로 사용해 요약에 필요한 기준을 고른다.
    label = item["classification"]["label"]
    return {**item, "policy": POLICIES[label], "prompt_version": PROMPT_VERSION,
            "prompt": build_prompt(item["text"], label)}


def create_chain(classifier):
    def classify(text):
        text = text.strip()
        classification = predict_text(text, classifier)
        # 분류 결과만 반환하면 뒤 단계에서 원문을 잃으므로 함께 전달한다.
        return {"text": text, "classification": classification}

    # 왼쪽 함수가 반환한 딕셔너리가 오른쪽 함수의 인자가 된다.
    return RunnableLambda(classify) | RunnableLambda(apply_policy)


def generate_prepared(prepared, provider):
    # 체인에서 만든 프롬프트 문자열을 그대로 생성 모델에 보낸다.
    return call_model(provider, prepared["prompt"])


def compare_prepared(prepared):
    # 같은 분류 결과와 프롬프트를 두 제공자가 공유한다.
    return {
        "classification": prepared["classification"],
        "results": [generate_prepared(prepared, name) for name in ("ollama", "openai")],
    }


def main():
    classifier = load_classifier()
    chain = create_chain(classifier)
    prepared = chain.invoke(INPUT_TEXT)
    print("분류 결과:", json.dumps(prepared["classification"], ensure_ascii=False))
    print("선택한 정책:", prepared["policy"])
    print("생성에 전달할 프롬프트:\n", prepared["prompt"])
    result = generate_prepared(prepared, PROVIDER)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["error"]:
        raise SystemExit("생성 호출이 실패했습니다. 연결 설정과 error를 확인하세요.")


if __name__ == "__main__":
    main()
