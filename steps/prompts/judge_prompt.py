"""요약 품질 채점용 체크리스트와 LLM 채점 프롬프트.

사람과 LLM이 같은 체크리스트를 채우고, 점수는 common/quality.py가 같은 공식으로 계산한다.
채점 기준(required_points, forbidden_claims)은 채점 단계에서만 쓰고 생성 프롬프트에는 넣지 않는다.
"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from prompts.document_policies import POLICIES

JUDGE_VERSION = "summary_checklist_v1"
POINT_STATUS = ("included", "partial", "missing")


class PointCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    status: Literal["included", "partial", "missing"]


class ClaimCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    violated: bool


class Checklist(BaseModel):
    """채점 결과. 사람 시트의 칸과 1:1로 대응한다."""

    model_config = ConfigDict(extra="forbid")
    required_points: list[PointCheck]
    forbidden_claims: list[ClaimCheck]
    unsupported_claims: list[str]  # forbidden_claims 외에 원문에 근거가 없는 주장(그대로 인용)
    numeric_error: bool  # 수치·날짜·시간·장소·담당자가 원문과 다름
    foreign_language: bool  # 한국어 문장에 외국어 문장이나 단어가 불필요하게 섞임
    policy_violation: bool  # 제안을 결정으로 씀, 미기재 처리 오류, 엉뚱한 필드에 넣음 등
    awkward_sentence: bool  # 문법이 틀리거나 뜻이 잘 통하지 않는 문장


def build_judge_prompt(source_text, label, summary, required_points, forbidden_claims):
    points = "\n".join(f"{i}. {point}" for i, point in enumerate(required_points, start=1))
    claims = "\n".join(f"{i}. {claim}" for i, claim in enumerate(forbidden_claims, start=1))
    return (
        "당신은 한국어 문서 요약의 채점자다. 원문과 요약을 비교해 체크리스트를 채운다.\n"
        "같은 뜻의 다른 표현은 인정한다. 미정과 확정을 구분한다. 원문을 기준으로만 판단한다.\n"
        f"문서 유형: {label}\n요약 정책: {POLICIES[label]}\n"
        "판정 기준\n"
        "- required_points: 각 핵심 내용이 요약에 있으면 included, 일부만 있으면 partial, "
        "없으면 missing\n"
        "- forbidden_claims: 요약이 해당 주장을 하면 violated=true\n"
        "- unsupported_claims: forbidden_claims 외에 원문에 근거가 없는 주장을 요약에서 그대로 "
        "인용해 나열. 없으면 []\n"
        "- numeric_error: 수치, 날짜, 시간, 장소, 담당자가 원문과 다르면 true\n"
        "- foreign_language: 한국어 대신 외국어 문장이나 불필요한 외국어 단어가 섞이면 true "
        "(원문에 있는 고유명사·약어는 제외)\n"
        "- policy_violation: 제안을 확정 결정으로 쓰거나, 원문에 있는 정보를 미기재로 두거나, "
        "내용을 엉뚱한 필드에 넣었으면 true\n"
        "- awkward_sentence: 문법이 틀리거나 뜻이 잘 통하지 않는 문장이 있으면 true\n"
        "출력은 Markdown 없이 아래 형식의 JSON 객체 하나다.\n"
        '{"required_points": [{"index": 1, "status": "included"}], '
        '"forbidden_claims": [{"index": 1, "violated": false}], "unsupported_claims": [], '
        '"numeric_error": false, "foreign_language": false, "policy_violation": false, '
        '"awkward_sentence": false}\n'
        f"[원문]\n{source_text}\n"
        f"[핵심 내용]\n{points}\n"
        f"[금지 주장]\n{claims}\n"
        f"[요약(JSON)]\n{json.dumps(summary, ensure_ascii=False, indent=1)}"
    )
