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
CLASSIFIER_MODEL = "klue/bert-base" # 기본 모델
# CLASSIFIER_MODEL = "klue/roberta-base"
# CLASSIFIER_MODEL = "monologg/koelectra-small-v3-discriminator"
# CLASSIFIER_MODEL = "monologg/koelectra-base-v3-discriminator"
# CLASSIFIER_MODEL = "monologg/distilkobert"
# CLASSIFIER_MODEL = "team-lucid/deberta-v3-base-korean"

# 주제와 학습자별 폴더에 학습 결과를 보관합니다.
OUTPUT_ROOT = ROOT / "artifacts"
LEARNER_DIR = OUTPUT_ROOT / DATASET / LEARNER
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
