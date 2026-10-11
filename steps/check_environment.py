"""uv가 사용하는 Python과 학습 준비 상태를 확인합니다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/check_environment.py
"""

import importlib.metadata
import os
import sys

import httpx
import torch
from lesson_settings import OLLAMA_URL, ROOT


def main():
    print("Python:", sys.executable)
    print("프로젝트:", ROOT)
    print("Python 버전:", sys.version.split()[0])
    for package in ("torch", "transformers", "scikit-learn", "fastapi", "langchain-core", "openai"):
        print(package + ":", importlib.metadata.version(package))
    print("GPU 사용 가능:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    # 키의 값은 표시하지 않고, 설정이 있는지만 확인합니다.
    print("OpenAI 키 설정:", bool(os.getenv("OPENAI_API_KEY")))
    try:
        response = httpx.get(OLLAMA_URL.rstrip("/") + "/api/tags", timeout=5, trust_env=False)
        response.raise_for_status()
        print("Ollama 모델:", [item["name"] for item in response.json().get("models", [])])
    except (httpx.HTTPError, ValueError, KeyError):
        print("Ollama: 실행 상태를 확인하세요. 데이터 확인과 분류 학습은 먼저 진행할 수 있습니다.")


if __name__ == "__main__":
    main()
