"""요약 품질 채점용 체크리스트와 LLM 채점 프롬프트.

사람과 LLM이 같은 체크리스트를 채우고, 점수는 common/quality.py가 같은 공식으로 계산한다.
채점 기준(required_points, forbidden_claims)은 채점 단계에서만 쓰고 생성 프롬프트에는 넣지 않는다.
"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from prompts.document_policies import FIELD_GUIDES, POLICIES, SCHEMAS

# v1 → v2: 생성 프롬프트와 같은 칸 규칙(FIELD_GUIDES)을 채점에도 주고, 정책 위반을 그 규칙 기준으로 판정.
#          문맥상 분명한 대상 풀어 쓰기는 근거 없는 주장으로 보지 않고, 결정·담당 겹침은 허용한다.
# v2 → v3: 정책 위반(예/아니오 하나)을 칸별 판정(field_checks)으로 바꿈. 칸 오류는 사용 가능 조건에 들어간다.
JUDGE_VERSION = "summary_checklist_v3"
POINT_STATUS = ("included", "partial", "missing")


def checked_fields(label):
    """칸별로 판정하는 칸. summary는 자유 서술이라 칸 판정에서 뺀다."""
    return [key for key in SCHEMAS[label].model_fields if key != "summary"]


class PointCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    status: Literal["included", "partial", "missing"]


class ClaimCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    violated: bool


class FieldCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    ok: bool
    reason: str  # 틀렸을 때 이유 한 줄. 맞으면 빈 문자열


class Checklist(BaseModel):
    """채점 결과. 사람 시트의 칸과 대응한다."""

    model_config = ConfigDict(extra="forbid")
    required_points: list[PointCheck]
    forbidden_claims: list[ClaimCheck]
    unsupported_claims: list[str]  # forbidden_claims 외에 원문에 근거가 없는 주장(그대로 인용)
    numeric_error: bool  # 수치·날짜·시간·장소·담당자가 원문과 다름
    foreign_language: bool  # 한국어 문장에 외국어 문장이나 단어가 불필요하게 섞임
    field_checks: list[FieldCheck]  # summary를 뺀 칸마다 칸 규칙대로 채웠는지
    awkward_sentence: bool  # 문법이 틀리거나 뜻이 잘 통하지 않는 문장


def build_judge_prompt(source_text, label, summary, required_points, forbidden_claims):
    points = "\n".join(f"{i}. {point}" for i, point in enumerate(required_points, start=1))
    claims = "\n".join(f"{i}. {claim}" for i, claim in enumerate(forbidden_claims, start=1))
    rules = "\n".join(f"  - {key}: {guide}" for key, guide in FIELD_GUIDES[label].items())
    fields = checked_fields(label)
    example_fields = ", ".join(f'{{"field": "{key}", "ok": true, "reason": ""}}' for key in fields)
    return (
        "당신은 한국어 문서 요약의 채점자다. 원문과 요약을 비교해 체크리스트를 채운다.\n"
        "같은 뜻의 다른 표현은 인정한다. 미정과 확정을 구분한다. 원문을 기준으로만 판단한다.\n"
        f"문서 유형: {label}\n요약 정책: {POLICIES[label]}\n"
        f"칸 규칙(요약은 이 규칙대로 써야 한다):\n{rules}\n"
        "null, [], '미기재'는 모두 '원문에 해당 정보 없음'이라는 같은 뜻이다.\n"
        "판정 기준\n"
        "- required_points: 각 핵심 내용이 요약 어딘가(어느 칸이든)에 있으면 included, 일부만 있으면 "
        "partial, 없으면 missing\n"
        "- forbidden_claims: 요약이 해당 주장을 하면 violated=true\n"
        "- unsupported_claims: forbidden_claims 외에 원문에 근거가 없는 주장을 요약에서 그대로 "
        "인용해 나열. 원문의 지칭이 가리키는 대상을 문맥상 분명한 범위에서 풀어 쓴 것은 "
        "근거 없는 주장이 아니다. 없으면 []\n"
        "- numeric_error: 수치, 날짜, 시간, 장소, 담당자가 원문과 다르면 true\n"
        "- foreign_language: 한국어 대신 외국어 문장이나 불필요한 외국어 단어가 섞이면 true "
        "(원문에 있는 고유명사·약어는 제외)\n"
        f"- field_checks: {', '.join(fields)} 칸마다 하나씩, 위 칸 규칙대로 채웠으면 ok=true. "
        "틀린 경우 ok=false와 이유 한 줄. 틀림의 예: 규칙상 다른 칸에 들어갈 내용이 있음, "
        "원문에 있는 해당 정보를 null·[]·미기재로 둠, 원문에 없는데 채움, 아직 정하지 않은 것을 "
        "결정으로 씀. 회의록에서 결정과 담당이 겹치는 것과, 맞는 칸이 없는 정보를 summary에 쓴 것은 "
        "틀림이 아니다\n"
        "- awkward_sentence: 문법이 틀리거나 뜻이 잘 통하지 않는 문장이 있으면 true\n"
        "출력은 Markdown 없이 아래 형식의 JSON 객체 하나다.\n"
        '{"required_points": [{"index": 1, "status": "included"}], '
        '"forbidden_claims": [{"index": 1, "violated": false}], "unsupported_claims": [], '
        '"numeric_error": false, "foreign_language": false, '
        f'"field_checks": [{example_fields}], "awkward_sentence": false}}\n'
        f"[원문]\n{source_text}\n"
        f"[핵심 내용]\n{points}\n"
        f"[금지 주장]\n{claims}\n"
        f"[요약(JSON)]\n{json.dumps(summary, ensure_ascii=False, indent=1)}"
    )
