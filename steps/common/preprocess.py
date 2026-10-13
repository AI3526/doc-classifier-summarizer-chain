"""긴 문서 전처리: 참고 자료를 떼어 내고 이번 본문만 남긴다.

심화 데이터(documents_advanced)의 긴 문서는 "이전 운영의 보관 자료"(다른 유형의 참고 문서)와
"이번에 전달하는 본문"이 섞여 있다. 라벨은 이번 본문의 유형을 따른다.
텍스트 안의 표시만 쓰며, 표시가 없는 문서는 그대로 돌려준다.
"""

import re

REFERENCE_HEADER = "이전 운영의 보관 자료"
REFERENCE_END = "이 내용은 당시 운영을 마치고 보관한 참고 자료입니다."
BODY_MARKERS = re.compile(r"이번에 전달하는 본문은 다음과 같습니다\.?|이번에 전달하는 본문")


def document_layout(text):
    """문서 구조: short(표시 없음) / reference_first / body_first."""
    if REFERENCE_HEADER not in text:
        return "short"
    return "reference_first" if text.lstrip().startswith(REFERENCE_HEADER) else "body_first"


def extract_body(text):
    """참고 자료 구간(머리말 앞 안내 문장 ~ 끝 표시)을 지우고 본문 표시 문구를 뺀다."""
    if REFERENCE_HEADER not in text or REFERENCE_END not in text:
        return text
    start = text.index(REFERENCE_HEADER)
    end = text.index(REFERENCE_END) + len(REFERENCE_END)
    if start > 0:
        # 본문 뒤에 붙은 참고 자료는 바로 앞 줄에 안내 문장이 있다. 그 줄부터 지운다.
        start = text.rfind("\n", 0, start - 1) + 1
    body = BODY_MARKERS.sub("", text[:start] + text[end:])
    return body.strip()
