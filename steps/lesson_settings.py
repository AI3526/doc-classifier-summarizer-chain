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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M") # 모델 더 비교?
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
TIMEOUT_SECONDS = float(os.getenv("GENERATION_TIMEOUT_SECONDS", "120"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))
