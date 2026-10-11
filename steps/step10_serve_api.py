"""10. 학습한 분류기와 생성 체인을 FastAPI 요청으로 실행한다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step10_serve_api.py
"""

from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from lesson_settings import GENERATION_CANDIDATES, GENERATION_DEFAULT
from pydantic import BaseModel, Field, field_validator
from step06_predict import load_classifier
from step08_connect_chain import (
    MAX_INPUT_CHARS,
    compare_prepared,
    create_chain,
    generate_prepared,
)

TargetName = Literal[tuple(GENERATION_CANDIDATES)]

HOST = "127.0.0.1"
PORT = 8000


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)

    @field_validator("text")
    @classmethod
    def nonblank(cls, text):
        if not text.strip():
            raise ValueError("공백만 입력할 수 없습니다.")
        return text.strip()


class GenerateRequest(TextRequest):
    target: TargetName = GENERATION_DEFAULT


class CompareRequest(TextRequest):
    targets: list[TargetName] = Field(default_factory=lambda: [GENERATION_DEFAULT], min_length=1)


@asynccontextmanager
async def lifespan(app):
    # 서버가 시작될 때 한 번 읽는다. 모델을 읽지 못하면 시작이 실패한다.
    classifier = load_classifier()
    app.state.chain = create_chain(classifier)
    yield


app = FastAPI(title="2차 프로젝트 단계별 API", lifespan=lifespan)


@app.get("/health")
def health():
    # 이 응답은 서버 동작만 확인한다. 생성 모델 연결을 시험하지 않는다.
    return {"status": "ok"}


@app.post("/generate")
def generate(body: GenerateRequest, request: Request):
    # 요청 검증에 실패하면 함수 실행 전 422를 반환한다.
    prepared = request.app.state.chain.invoke(body.text)
    result = generate_prepared(prepared, body.target)
    response = {"classification": prepared["classification"], "result": result}
    # 200이어도 답변의 완료 여부, 형식, 내용은 별도로 확인해야 한다.
    return JSONResponse(response, status_code=502 if result["error"] else 200)


@app.post("/compare")
def compare(body: CompareRequest, request: Request):
    # 한 번 분류해서 만든 프롬프트를 여러 생성 모델이 공유한다.
    prepared = request.app.state.chain.invoke(body.text)
    result = compare_prepared(prepared, body.targets)
    failed = any(row["error"] for row in result["results"])
    return JSONResponse(result, status_code=502 if failed else 200)


def main():
    print(f"API 문서: http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT, access_log=False)


if __name__ == "__main__":
    main()
