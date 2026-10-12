"""단계별 실습에서 함께 쓰는 경로와 설정. 비밀 키는 .env에 보관합니다."""

import os
from pathlib import Path

from dotenv import load_dotenv

# 이 파일이 있는 steps 폴더의 한 단계 위가 프로젝트 폴더입니다.
# 터미널의 현재 위치가 달라도 같은 데이터와 결과 폴더를 사용합니다.
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

# 필수 실험은 documents, 긴 문서 선택 실험은 documents_advanced를 사용합니다.
DATASET = "documents"
# LEARNER는 "learner_park", "learner_choi", "learner_ro", "learner_su" 중 하나
LEARNER = "learner_park" 
# 사전학습 분류 후보. 왼쪽 짧은 이름으로 step04에 넘기고 실행 폴더 이름에도 씁니다.
# 예: uv run python steps/step04_train_classifier.py --model bert roberta --lr 1e-5 2e-5 5e-5
CLASSIFIER_CANDIDATES = {
    "bert": "klue/bert-base",  # 필수 후보
    "roberta": "klue/roberta-base",
    "electra-s": "monologg/koelectra-small-v3-discriminator",  # 14M, 작고 빠른 후보
    "electra-b": "monologg/koelectra-base-v3-discriminator",
    "deberta-ko": "team-lucid/deberta-v3-base-korean",  # 한국어 전용 DeBERTa-v3
    # "mdeberta": "microsoft/mdeberta-v3-base",  # 다국어. sentencepiece, protobuf 설치 필요
}
CLASSIFIER_MODEL = "bert"  # 기본 모델

# 주제와 학습자별 폴더에 학습 결과를 보관합니다.
OUTPUT_ROOT = ROOT / "artifacts"
LEARNER_DIR = OUTPUT_ROOT / DATASET / LEARNER
# 팀 전체 실행 중 서비스에 쓸 분류기 하나를 고른 결과와 test 평가를 보관합니다.
SELECTION_DIR = OUTPUT_ROOT / DATASET / "selection"
DATA_DIR = ROOT / "data" / DATASET
# 생성 코드가 동작하는지만 확인하는 2건짜리 입력. 개발 비교는 DATA_DIR의 15건을 사용합니다.
DEVELOPMENT_INPUT = ROOT / "steps" / "examples" / "documents_development.jsonl"

# OpenAI 키는 읽어서 클라이언트에 전달하며 화면이나 결과 파일에 출력하지 않습니다.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
TIMEOUT_SECONDS = float(os.getenv("GENERATION_TIMEOUT_SECONDS", "120"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))

# 생성 모델 후보. 왼쪽 짧은 이름으로 step07~10에 넘깁니다. 모든 후보에 같은 프롬프트와 설정을 씁니다.
GENERATION_CANDIDATES = {
    "qwen3-4b": {"provider": "ollama", "model": OLLAMA_MODEL},
    "gemma3-4b": {"provider": "ollama", "model": "gemma3:4b"},
    "qwen2.5-7b": {"provider": "ollama", "model": "qwen2.5:7b"},
    "exaone3.5-7.8b": {"provider": "ollama", "model": "exaone3.5:7.8b"},
    "llama3.1-8b": {"provider": "ollama", "model": "llama3.1:8b"},  # 한국어 약세 대조군
    "openai": {"provider": "openai", "model": OPENAI_MODEL},  # 요금 발생
}
GENERATION_DEFAULT = "qwen3-4b"  # 생성 모델 비교 전 서비스 기본값
# 구조화 출력(스키마에 맞는 토큰만 생성) 사용 여부. 1차 비교 결과로 서비스 기본값을 사용으로 정했다.
# step09는 --structured로 따로 정한다(2차 비교는 사용).
STRUCTURED_OUTPUT = True
# 하이브리드 라우팅: 기밀 문서는 로컬 모델로만 보낸다(외부 전송 금지).
# 생성 모델이 연결 실패·시간 초과면 대체 모델로 넘기되, 기밀 문서는 로컬 모델 사이에서만 넘긴다.
CONFIDENTIAL_TARGET = "qwen2.5-7b"  # 최종 비교에서 환각이 가장 적고 읽기용 기준을 통과한 로컬 모델
LOCAL_FALLBACK_TARGET = "qwen3-4b"  # 로컬 모델의 대체(최종 비교의 두 번째 로컬 후보)
# 요약 품질 LLM 채점 모델. 후보이기도 해서 자기 답변을 후하게 볼 수 있음을 한계로 적는다.
JUDGE_TARGET = "openai"
TEMPERATURE = 0.2
OLLAMA_NUM_CTX = 4096
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "none")

def _optional_float(name, default=None):
    value = os.getenv(name, "").strip()
    return float(value) if value else default


# OpenAI 비용 계산용 단가(USD, 100만 토큰당). gpt-6-luna Standard 처리 기준, 2026-10-11 확인.
# - 캐시된 입력은 입력 단가의 10%, 캐시 쓰기는 1.25배
# - 한 요청의 입력이 272K 토큰을 넘으면 요청 전체에 입력·캐시 2배, 출력 1.5배
# - 지역 처리 +10%, Batch·Flex 50%, Fast 2배는 이 프로젝트에서 쓰지 않으므로 반영하지 않음
# 모델이나 가격이 바뀌면 .env에서 같은 이름으로 덮어씁니다.
OPENAI_INPUT_PRICE_PER_1M = _optional_float("OPENAI_INPUT_PRICE_PER_1M", 0.10)
OPENAI_CACHED_INPUT_PRICE_PER_1M = _optional_float("OPENAI_CACHED_INPUT_PRICE_PER_1M", 0.01)
OPENAI_CACHE_WRITE_PRICE_PER_1M = _optional_float("OPENAI_CACHE_WRITE_PRICE_PER_1M", 0.125)
OPENAI_OUTPUT_PRICE_PER_1M = _optional_float("OPENAI_OUTPUT_PRICE_PER_1M", 0.50)
OPENAI_LONG_PROMPT_TOKENS = 272_000
OPENAI_LONG_PROMPT_MULTIPLIER = {"input": 2.0, "output": 1.5}
OPENAI_PRICE_CHECKED_AT = os.getenv("OPENAI_PRICE_CHECKED_AT", "2026-10-11")
# 로컬 추론 전력 추정용 소비 전력(W). 비워 두면 전력량을 추정하지 않습니다.
LOCAL_POWER_WATTS = _optional_float("LOCAL_POWER_WATTS")
