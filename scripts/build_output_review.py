"""생성 비교 폴더 하나를 사람이 읽기 쉬운 문서로 만든다: 원문 → 채점 기준 → 모델별 칸 내용과 LLM 채점.

사람 블라인드 채점이 남은 폴더(human/ 시트가 있고 quality_human.json이 없음)는 모델 이름과
LLM 판정이 보이면 블라인드가 깨지므로 기본적으로 만들지 않는다(--allow-unblinded로 강제).

uv run python scripts/build_output_review.py --dir artifacts/documents_advanced/learner_park/comparisons/<폴더>
결과: <폴더>/output_review.md
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

from step09b_score import latest_judgements

LABEL_KO = {"announcement": "공지", "manual": "매뉴얼", "minutes": "회의록"}
FIELD_KO = {"summary": "요약", "audience": "대상", "schedule": "일정", "actions": "대상자가 할 일",
            "changes": "달라진 점", "prerequisites": "사전 준비", "steps": "절차",
            "cautions": "주의 사항", "discussion": "논의", "decisions": "결정", "owners": "담당",
            "follow_ups": "미정 사항"}
POINT_KO = {"included": "포함", "partial": "부분", "missing": "누락"}


def cell(value):
    if value is None:
        return "*(null)*"
    if isinstance(value, list):
        return "<br>".join(f"• {item}" for item in value) if value else "*(빈 목록)*"
    return str(value).replace("|", "\\|").replace("\n", " ")


def model_block(record, judge):
    parsed = record["result"]["parsed"]
    lines = [f"#### {record['model_info']['alias']} (반복 {record['repeat']})", ""]
    if not isinstance(parsed, dict):
        return lines + [f"형식 실패: {record['result']['format_error']}", ""]
    checks = {c["field"]: c for c in (judge or {}).get("checklist", {}).get("field_checks", [])}
    lines += ["| 칸 | 내용 | 칸 판정 |", "|---|---|---|"]
    for field, value in parsed.items():
        check = checks.get(field)
        verdict = "–" if check is None else ("맞음" if check["ok"] else f"**틀림**: {check['reason']}")
        lines.append(f"| {FIELD_KO.get(field, field)} | {cell(value)} | {verdict} |")
    if judge and judge.get("metrics"):
        metrics, checklist = judge["metrics"], judge["checklist"]
        points = ", ".join(f"{p['index']}번 {POINT_KO.get(p['status'], p['status'])}"
                           for p in checklist["required_points"] if p["status"] != "included")
        claims = ", ".join(f"{c['index']}번" for c in checklist["forbidden_claims"] if c["violated"])
        lines += ["", (f"- 사용 가능: **{'O' if metrics['usable'] else 'X'}** · 읽기용: "
                      f"{'O' if metrics['usable_reading'] else 'X'} · 핵심 포함률 {metrics['coverage']:.2f}"
                      f" · 감점 점수 {metrics['deduction_score']}"),
                  f"- 빠진 필수 포인트: {points or '없음'} · 어긴 금지 주장: {claims or '없음'}",
                  (f"- 근거 없는 주장: {'; '.join(checklist['unsupported_claims']) or '없음'}"
                  f" · 수치 오류: {'있음' if checklist['numeric_error'] else '없음'}"
                  f" · 외국어: {'있음' if metrics['foreign_language'] else '없음'}")]
    elif judge is None:
        lines += ["", "- LLM 채점 없음"]
    return lines + [""]


def main():
    parser = argparse.ArgumentParser(description="생성 결과 읽기용 문서")
    parser.add_argument("--dir", type=Path, required=True, help="생성 비교 폴더")
    parser.add_argument("--repeat", type=int, default=1, help="몇 번째 반복만 볼지(0이면 전부)")
    parser.add_argument("--allow-unblinded", action="store_true",
                        help="사람 블라인드 채점이 끝나지 않은 폴더도 만든다")
    args = parser.parse_args()
    folder = args.dir
    if ((folder / "human").is_dir() and any((folder / "human").glob("*.csv"))
            and not (folder / "quality_human.json").exists() and not args.allow_unblinded):
        raise SystemExit("사람 블라인드 채점이 끝나지 않은 폴더입니다. 채점 후에 만들거나 "
                         "--allow-unblinded를 붙이세요.")
    records = [json.loads(line) for line in (folder / "results.jsonl").open(encoding="utf-8")
               if line.strip()]
    if args.repeat:
        records = [r for r in records if r["repeat"] == args.repeat]
    judges = latest_judgements(folder) if (folder / "judge.jsonl").exists() else {}
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))

    cases = {}
    for record in records:
        cases.setdefault(record["case_id"], []).append(record)
    lines = [f"# 생성 결과 보기: {folder.name}", "",
             (f"`scripts/build_output_review.py`가 생성. 프롬프트 {config.get('prompt_version')}, "
             f"구조화 출력 {config.get('structured_output')}, "
             f"{'반복 ' + str(args.repeat) + '회차만' if args.repeat else '모든 반복'}."), "",
             "채점 기준(필수 포인트, 금지 주장)은 채점에만 쓰였고 프롬프트에는 들어가지 않았다.", ""]
    for number, (case_id, rows) in enumerate(cases.items(), start=1):
        first = rows[0]
        expected = first["scoring_reference"].get("expected_label")
        predicted = first["input_data"]["predicted_label"]
        mark = "" if not expected or expected == predicted else " ⚠ 오분류"
        lines += [f"## {number}. {case_id}", "",
                  (f"정답 유형 **{LABEL_KO.get(expected, expected)}** · 예측 유형 "
                  f"**{LABEL_KO.get(predicted, predicted)}** (확신도 "
                  f"{first['input_data']['classification_confidence']:.2f}){mark}"), "",
                  "**원문**", ""]
        lines += [f"> {line}" if line.strip() else ">" for line in
                  first["input_data"]["source_text"].split("\n")]
        reference = first["scoring_reference"]
        if reference.get("required_points"):
            lines += ["", "**필수 포인트**", ""] + [f"{i}. {p}" for i, p in
                                                   enumerate(reference["required_points"], 1)]
        if reference.get("forbidden_claims"):
            lines += ["", "**금지 주장**", ""] + [f"{i}. {c}" for i, c in
                                                enumerate(reference["forbidden_claims"], 1)]
        lines.append("")
        for record in rows:
            lines += model_block(record, judges.get(record["experiment_id"]))
    out = folder / "output_review.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("저장:", out, f"({len(cases)}건, 응답 {len(records)}개)")


if __name__ == "__main__":
    main()
