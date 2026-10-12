"""생성 프롬프트에 넣는 퓨샷 예시. 문서 유형마다 2건, 프롬프트에는 예측 유형의 2건만 넣는다.

예시 문서는 data/documents/train.csv에서 골랐다. 개발·평가 데이터에서 고르면 그 데이터로 잰 점수가
오염되기 때문이다. 원문은 id로 train.csv에서 읽어 데이터와 어긋나지 않게 하고, 모범 출력은 팀이 검토해
확정한 것이다(2026-10-12). 출력을 바꾸면 document_policies.PROMPT_VERSION도 바꾼다.
"""

import csv
from functools import cache
from pathlib import Path

TRAIN_CSV = Path(__file__).resolve().parents[2] / "data" / "documents" / "train.csv"

EXAMPLES = {
    "announcement": [
        # 기관이 한 일만 있어 대상자 행동은 없고, 물품이 보관함으로 '이동'한 상태 변화가 있다.
        ("documents-v2-announcement-train-009-v1", {
            "summary": ("이번 주 도착한 소형 택배 중 수령인이 확인된 물품은 보관함으로 옮겼고, "
                        "이름을 확인하지 못한 물품은 안내실에서 따로 보관한다."),
            "audience": "이번 주 소형 택배 수령 대상자",
            "schedule": "이번 주",
            "actions": [],
            "changes": "수령인이 확인된 이번 주 소형 택배가 보관함으로 이동했다.",
        }),
        # 추가 모집은 변화가 아니고, 미정인 일정도 schedule에 그대로 쓴다.
        ("documents-v2-announcement-train-054-v1", {
            "summary": ("어린이 탐방 활동의 동행 봉사자를 추가 모집하며, 사전 안전 교육에 참여할 수 있는 "
                        "사람만 신청할 수 있다."),
            "audience": "어린이 탐방 활동 동행 봉사자 지원자",
            "schedule": "교육 일정 미정(신청자 수 확인 후 확정)",
            "actions": ["사전 안전 교육에 참여할 수 있는 경우에만 신청한다."],
            "changes": None,
        }),
    ],
    "manual": [
        # "먼저/사전에" 같은 표시가 없어 전제 조건은 비우고, 상황은 summary에 쓴다.
        ("documents-v2-manual-train-042-v1", {
            "summary": "운송 목록에 없는 물품을 발견했을 때 소유를 확인하는 절차다.",
            "prerequisites": [],
            "steps": ["물품 사진과 발견 위치를 기록한다.", "기록을 발송 담당에게 보내 소유를 확인한다."],
            "cautions": ["목록에 즉시 추가 확정하지 않는다.", "소유가 확인될 때까지 다른 팀에 넘기지 않는다."],
        }),
        # "먼저" 확인하는 일은 전제 조건으로 두고 steps에서 반복하지 않는다.
        ("documents-v2-manual-train-014-v1", {
            "summary": "전시 받침을 고르는 절차다.",
            "prerequisites": ["작품의 바닥 크기와 무게를 확인한다."],
            "steps": ["받침 위에서 작품이 흔들리지 않는지 점검한다.",
                      "안정적인 받침을 찾지 못하면 담당자에게 문의한다."],
            "cautions": ["안정적인 받침이 없어도 임시 상자로 대신하지 않는다."],
        }),
    ],
    "minutes": [
        # 확정한 방침은 결정, '다음에 정한다'는 후속 조치.
        ("documents-v2-minutes-train-043-v1", {
            "summary": ("늦게 끝나는 저녁 행사의 부담을 논의했고, 발표 수를 줄이기 전에 쉬는 시간과 "
                        "이동 시간을 다시 계산하기로 했다. 종료 시각 변경은 다음 검토에서 정한다."),
            "discussion": ["늦게 끝나는 행사에 대한 참가자의 부담"],
            "decisions": ["발표 수를 줄이기 전에 쉬는 시간과 이동 시간을 다시 계산한다."],
            "owners": [],
            "follow_ups": ["종료 시각 변경 여부를 다음 검토에서 결정한다."],
        }),
        # '하기로 했다'는 모두 결정에 넣고, 담당자가 있는 일은 담당에도 적는다(겹침 허용).
        ("documents-v2-minutes-train-028-v1", {
            "summary": "안내서가 많이 남은 원인을 검토하고, 다음 부수는 실제 안내서 수령 비율로 계산하기로 했다.",
            "discussion": ["안내서가 많이 남은 원인"],
            "decisions": ["다음 안내서 부수는 방문자 수 대신 실제 수령 비율로 계산한다.",
                          "최근 세 행사의 기록을 모은다."],
            "owners": ["인쇄 담당: 최근 세 행사의 기록 수집 (마감 미기재)"],
            "follow_ups": [],
        }),
    ],
}


@cache
def train_texts():
    with TRAIN_CSV.open(encoding="utf-8-sig", newline="") as stream:
        return {row["id"]: row["text"] for row in csv.DictReader(stream)}


def examples_for(label):
    """(원문, 모범 출력) 목록. 예시 id가 train.csv에 없으면 바로 알 수 있게 KeyError를 낸다."""
    texts = train_texts()
    return [(texts[example_id], output) for example_id, output in EXAMPLES[label]]
