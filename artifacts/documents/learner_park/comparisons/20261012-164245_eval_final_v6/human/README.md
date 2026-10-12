# 사람 채점 안내

- 항목 36개, 모델 이름은 가려져 있습니다. `human_key.json`은 채점이 끝날 때까지 열지 마세요.
- 자기 번호의 `human_sheet_raterN.csv`만 채웁니다. 다른 채점자와 상의하지 않습니다.
- 원문(source_text)만 근거로 판단합니다. 같은 뜻의 다른 표현은 인정하고, 미정과 확정을 구분합니다.

| 칸 | 입력 | 기준 |
|---|---|---|
| rp1~rp3 | 포함 / 부분 / 누락 | 핵심 내용(required_points) 각 번호가 요약에 있는지 |
| fc1~fc2 | 1 / 0 | 금지 주장(forbidden_claims) 각 번호를 요약이 하면 1 |
| unsupported_claims | 문장 인용, 없으면 빈칸 | 금지 주장 외에 원문에 근거가 없는 내용 |
| numeric_error | 1 / 0 | 수치·날짜·시간·장소·담당자가 원문과 다름 |
| foreign_language | 1 / 0 | 외국어 문장이나 불필요한 외국어 단어가 섞임(원문의 고유명사·약어 제외) |
| wrong_fields | 칸 이름, 쉼표로 구분. 모두 맞으면 빈칸 | checked_fields에 적힌 칸 중 칸 규칙대로 채우지 않은 칸(예: `actions, changes`). 규칙은 `reports/evaluation_rubric.md`의 칸 규칙 표 |
| wrong_field_reasons | 자유 | 틀린 칸마다 이유 한 줄 |
| awkward_sentence | 1 / 0 | 문법이 틀리거나 뜻이 잘 통하지 않는 문장 |
| notes | 자유 | 판단 근거나 애매했던 점 |

채점 전에 개발 데이터 3~5건으로 함께 연습해 기준을 맞추세요. 평가 데이터로 연습하지 않습니다.
