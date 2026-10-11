"""09b. 생성 비교 결과의 요약 품질을 채점한다. 사람과 LLM이 같은 체크리스트를 쓴다.

1) judge: LLM이 모든 응답을 채점한다(openai, 요금 발생). 중간에 멈춰도 이어서 채점한다.
2) sheet: 모델명을 가린 사람 채점 시트를 만든다. 모델·문서 유형별로 고르게 36건, 채점자마다 순서를 섞는다.
3) aggregate: 채운 시트를 모아 다수결, 채점자 간 일치도, LLM과의 일치도를 계산한다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step09b_score.py judge                 # 가장 최근 비교 폴더
uv run python steps/step09b_score.py sheet --dir <비교 폴더>
uv run python steps/step09b_score.py aggregate --dir <비교 폴더>
uv run python steps/step09b_score.py aggregate --dummy        # 가상 채점(미리보기) 집계
uv run python steps/step09b_score.py select [--dummy]         # 서비스 생성 모델 선택(평가 기준 8장)

--dummy는 human_dummy/의 가상 채점 시트를 집계해 quality_human_dummy.json에 is_dummy=true로 남긴다.
실제 사람 채점 결과(quality_human.json)와 섞이지 않는다.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import lesson_settings as lesson
from common.quality import (
    BINARY_FIELDS,
    choose_generation_model,
    cohen_kappa,
    detect_foreign,
    fleiss_kappa,
    llm_trust_decision,
    pabak,
    pearson,
    score_checklist,
    summarize_scores,
)
from common.records import now_iso, write_json
from prompts.judge_prompt import JUDGE_VERSION, Checklist, build_judge_prompt
from pydantic import ValidationError
from step07_call_models import call_model

HUMAN_ITEMS = 36
HUMAN_RATERS = 4
SAMPLE_SEED = 2026
STATUS_KO = {"포함": "included", "부분": "partial", "누락": "missing"}
FLAG_COLUMNS = ("numeric_error", "foreign_language", "policy_violation", "awkward_sentence")
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def latest_comparison():
    folders = sorted((lesson.LEARNER_DIR / "comparisons").glob("*/results.jsonl"))
    if not folders:
        raise FileNotFoundError("비교 결과가 없습니다. step09_compare.py를 먼저 실행하세요.")
    return folders[-1].parent


def read_records(directory):
    with (Path(directory) / "results.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parseable(record):
    return record["result"]["status"] == "SUCCESS" and record["result"]["format_valid_lenient"]


def foreign_auto(record):
    return detect_foreign(record["result"]["parsed"], record["input_data"]["source_text"])


# ---------- 1) LLM 채점 ----------

def parse_checklist(text, n_points, n_claims):
    match = _FENCE.match(text or "")
    checklist = Checklist.model_validate_json(match.group(1) if match else text).model_dump()
    if sorted(item["index"] for item in checklist["required_points"]) != list(range(1, n_points + 1)):
        raise ValueError("핵심 내용 판정이 채점 기준 개수와 맞지 않습니다.")
    claimed = {item["index"] for item in checklist["forbidden_claims"]}
    if not claimed <= set(range(1, n_claims + 1)):
        raise ValueError("금지 주장 번호가 채점 기준 범위를 벗어났습니다.")
    # 위반한 금지 주장만 나열하는 채점 모델이 있어, 빠진 번호는 위반 없음으로 채운다.
    checklist["forbidden_claims"] += [{"index": i, "violated": False}
                                      for i in range(1, n_claims + 1) if i not in claimed]
    checklist["forbidden_claims"].sort(key=lambda item: item["index"])
    return checklist


def judge_record(record, attempts=2):
    """응답 하나를 채점한다. 형식이 틀리면 한 번 더 시도하고, 두 호출의 비용을 모두 더한다."""
    reference = record["scoring_reference"]
    points, claims = reference["required_points"] or [], reference["forbidden_claims"] or []
    prompt = build_judge_prompt(record["input_data"]["source_text"],
                                record["input_data"]["predicted_label"],
                                record["result"]["parsed"], points, claims)
    cost, checklist, error, result = 0.0, None, None, None
    for _ in range(attempts):
        result = call_model(lesson.JUDGE_TARGET, prompt, temperature=0, max_tokens=800)
        cost += result["cost_metrics"]["api_cost_usd"] or 0
        error = result["error"]
        if result["status"] != "SUCCESS":
            continue
        try:
            checklist = parse_checklist(result["text"], len(points), len(claims))
            error = None
            break
        except (ValidationError, ValueError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
    return checklist, error, {**result, "total_cost_usd": cost}


def run_judge(directory):
    directory = Path(directory)
    records = read_records(directory)
    if not records[0]["scoring_reference"].get("required_points"):
        raise ValueError("이 비교 입력에는 채점 기준(required_points)이 없습니다.")
    output = directory / "judge.jsonl"
    done = set()
    if output.exists():
        done = {json.loads(line)["experiment_id"] for line in output.open(encoding="utf-8")
                if line.strip() and json.loads(line)["judge_status"] == "SUCCESS"}
    with output.open("a", encoding="utf-8") as stream:
        for index, record in enumerate(records, start=1):
            if record["experiment_id"] in done:
                continue
            row = {"experiment_id": record["experiment_id"],
                   "alias": record["model_info"]["alias"], "case_id": record["case_id"],
                   "repeat": record["repeat"], "judge_version": JUDGE_VERSION,
                   "judge_target": lesson.JUDGE_TARGET, "foreign_auto": None}
            if not parseable(record):
                # 형식이 깨진 응답은 채점하지 않고 0점, 사용 불가로 둔다.
                row.update(judge_status="SKIPPED_FORMAT", checklist=None, judge_error=None,
                           judge_cost_usd=0.0, metrics=score_checklist(None, format_ok=False))
            else:
                checklist, error, result = judge_record(record)
                row.update(foreign_auto=foreign_auto(record), checklist=checklist,
                           judge_status="SUCCESS" if checklist else "ERROR", judge_error=error,
                           judge_cost_usd=result["total_cost_usd"],
                           judge_raw=result["text"],
                           metrics=score_checklist(checklist, True, foreign_auto(record))
                           if checklist else None)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{index}/{len(records)} {row['alias']} {row['case_id']} r{row['repeat']}: "
                  f"{row['judge_status']}")
    return write_llm_quality(directory)


def latest_judgements(directory):
    rows = {}
    for line in (Path(directory) / "judge.jsonl").open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            rows[row["experiment_id"]] = row  # 같은 응답을 다시 채점했다면 마지막 결과를 쓴다.
    return rows


def write_llm_quality(directory):
    rows = latest_judgements(directory)
    scored = [row for row in rows.values() if row["metrics"] is not None]
    by_alias = {}
    for row in scored:
        by_alias.setdefault(row["alias"], []).append(row["metrics"])
    quality = {"judge_version": JUDGE_VERSION, "judge_target": lesson.JUDGE_TARGET,
               "judged": len(scored), "judge_errors": sum(r["judge_status"] == "ERROR"
                                                           for r in rows.values()),
               "judge_cost_total_usd": sum(r.get("judge_cost_usd") or 0 for r in rows.values()),
               "targets": {alias: summarize_scores(metrics) for alias, metrics in by_alias.items()},
               "updated_at": now_iso()}
    write_json(Path(directory) / "quality_llm.json", quality)
    return quality


# ---------- 2) 사람 채점 시트 ----------

def sample_items(records):
    """모델별로 같은 수, 그 안에서 문서 유형별로 같은 수를 무작위로 고른다(seed 고정)."""
    pool = [row for row in records if row["repeat"] == 1 and parseable(row)]
    aliases = sorted({row["model_info"]["alias"] for row in pool})
    labels = sorted({row["scoring_reference"]["expected_label"] for row in pool})
    per_cell = HUMAN_ITEMS // (len(aliases) * len(labels))
    rng = random.Random(SAMPLE_SEED)
    chosen = []
    for alias in aliases:
        for label in labels:
            cell = [row for row in pool if row["model_info"]["alias"] == alias
                    and row["scoring_reference"]["expected_label"] == label]
            chosen += rng.sample(cell, min(per_cell, len(cell)))
    return chosen


def make_sheets(directory):
    directory = Path(directory)
    items = sample_items(read_records(directory))
    n_points = max(len(row["scoring_reference"]["required_points"]) for row in items)
    n_claims = max(len(row["scoring_reference"]["forbidden_claims"]) for row in items)
    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(items)
    key = {f"H{i:02d}": row["experiment_id"] for i, row in enumerate(items, start=1)}
    columns = (["item_id", "document_type", "source_text", "summary", "required_points",
                "forbidden_claims"] + [f"rp{i}" for i in range(1, n_points + 1)]
               + [f"fc{i}" for i in range(1, n_claims + 1)]
               + ["unsupported_claims", *FLAG_COLUMNS, "notes"])
    sheet_dir = directory / "human"
    sheet_dir.mkdir(exist_ok=True)
    by_id = {row["experiment_id"]: row for row in items}
    for rater in range(1, HUMAN_RATERS + 1):
        order = list(key.items())
        random.Random(SAMPLE_SEED + rater).shuffle(order)  # 채점자마다 순서를 다르게 섞는다.
        with (sheet_dir / f"human_sheet_rater{rater}.csv").open(
                "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for item_id, experiment_id in order:
                row = by_id[experiment_id]
                reference = row["scoring_reference"]
                writer.writerow({
                    "item_id": item_id,
                    "document_type": row["input_data"]["predicted_label"],
                    "source_text": row["input_data"]["source_text"],
                    "summary": json.dumps(row["result"]["parsed"], ensure_ascii=False, indent=1),
                    "required_points": "\n".join(f"{i}. {p}" for i, p in
                                                 enumerate(reference["required_points"], 1)),
                    "forbidden_claims": "\n".join(f"{i}. {c}" for i, c in
                                                  enumerate(reference["forbidden_claims"], 1)),
                })
    # 정답 대조표는 채점이 끝날 때까지 열지 않는다.
    write_json(sheet_dir / "human_key.json",
               {"seed": SAMPLE_SEED, "items": key,
                "aliases": {item_id: by_id[eid]["model_info"]["alias"]
                            for item_id, eid in key.items()}})
    (sheet_dir / "README.md").write_text(GUIDE.format(n=len(items), points=n_points,
                                                      claims=n_claims), encoding="utf-8")
    print(f"시트 {HUMAN_RATERS}개, 항목 {len(items)}개:", sheet_dir)
    return sheet_dir


GUIDE = """# 사람 채점 안내

- 항목 {n}개, 모델 이름은 가려져 있습니다. `human_key.json`은 채점이 끝날 때까지 열지 마세요.
- 자기 번호의 `human_sheet_raterN.csv`만 채웁니다. 다른 채점자와 상의하지 않습니다.
- 원문(source_text)만 근거로 판단합니다. 같은 뜻의 다른 표현은 인정하고, 미정과 확정을 구분합니다.

| 칸 | 입력 | 기준 |
|---|---|---|
| rp1~rp{points} | 포함 / 부분 / 누락 | 핵심 내용(required_points) 각 번호가 요약에 있는지 |
| fc1~fc{claims} | 1 / 0 | 금지 주장(forbidden_claims) 각 번호를 요약이 하면 1 |
| unsupported_claims | 문장 인용, 없으면 빈칸 | 금지 주장 외에 원문에 근거가 없는 내용 |
| numeric_error | 1 / 0 | 수치·날짜·시간·장소·담당자가 원문과 다름 |
| foreign_language | 1 / 0 | 외국어 문장이나 불필요한 외국어 단어가 섞임(원문의 고유명사·약어 제외) |
| policy_violation | 1 / 0 | 제안을 결정으로 씀, 원문에 있는 정보를 미기재로 둠, 엉뚱한 칸에 넣음 |
| awkward_sentence | 1 / 0 | 문법이 틀리거나 뜻이 잘 통하지 않는 문장 |
| notes | 자유 | 판단 근거나 애매했던 점 |

채점 전에 개발 데이터 3~5건으로 함께 연습해 기준을 맞추세요. 평가 데이터로 연습하지 않습니다.
"""


# ---------- 3) 집계 ----------

def read_sheet(path, n_points, n_claims):
    """채운 시트를 체크리스트로 바꾼다. 비어 있거나 잘못된 칸이 있으면 알려 준다."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result, problems = {}, []
    fillable = [name for name in rows[0] if re.fullmatch(r"(rp|fc)\d+", name)] + list(FLAG_COLUMNS) \
        if rows else []
    for row in rows:
        item = row["item_id"]
        if not any(row[name].strip() for name in fillable):
            continue  # 아직 채우지 않은 행은 건너뛴다.
        try:
            points = [{"index": i, "status": STATUS_KO[row[f"rp{i}"].strip()]}
                      for i in range(1, n_points + 1) if f"rp{i}" in row and row[f"rp{i}"].strip()]
            claims = [{"index": i, "violated": row[f"fc{i}"].strip() == "1"}
                      for i in range(1, n_claims + 1) if f"fc{i}" in row and row[f"fc{i}"].strip()]
            flags = {name: row[name].strip() == "1" for name in FLAG_COLUMNS}
            if any(row[name].strip() not in ("0", "1") for name in FLAG_COLUMNS):
                raise KeyError("0/1")
        except KeyError as exc:
            problems.append(f"{Path(path).name} {item}: 잘못된 값 {exc}")
            continue
        unsupported = [row["unsupported_claims"].strip()] if row["unsupported_claims"].strip() else []
        result[item] = {"required_points": points, "forbidden_claims": claims,
                        "unsupported_claims": unsupported, **flags}
    return result, problems


def majority(values, tie):
    counts = Counter(values).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return tie, True
    return counts[0][0], False


def aggregate(directory, dummy=False):
    directory = Path(directory)
    sheet_dir = directory / ("human_dummy" if dummy else "human")
    key = json.loads((sheet_dir / "human_key.json").read_text(encoding="utf-8"))
    records = {row["experiment_id"]: row for row in read_records(directory)}
    references = [records[eid]["scoring_reference"] for eid in key["items"].values()]
    n_points = max(len(ref["required_points"]) for ref in references)
    n_claims = max(len(ref["forbidden_claims"]) for ref in references)
    sheets, problems = {}, []
    for path in sorted(sheet_dir.glob("human_sheet_rater*.csv")):
        filled, issues = read_sheet(path, n_points, n_claims)
        problems += issues
        if filled:
            sheets[path.stem.replace("human_sheet_", "")] = filled
    if len(sheets) < 2:
        raise ValueError("채운 시트가 2개 이상 있어야 합니다.")

    judgements = latest_judgements(directory) if (directory / "judge.jsonl").exists() else {}
    items, ties, rater_flags = [], [], {field: [] for field in BINARY_FIELDS}
    for item_id, experiment_id in key["items"].items():
        record = records[experiment_id]
        expected = (len(record["scoring_reference"]["required_points"]),
                    len(record["scoring_reference"]["forbidden_claims"]))
        per_rater = []
        for rater, sheet in sheets.items():
            check = sheet.get(item_id)
            if check and (len(check["required_points"]), len(check["forbidden_claims"])) == expected:
                per_rater.append(check)
            else:
                problems.append(f"{rater} {item_id}: 비어 있거나 rp/fc 칸 수가 맞지 않아 제외")
        if len(per_rater) < 2:
            problems.append(f"{item_id}: 유효한 채점이 2개 미만이라 집계에서 제외")
            continue
        scored = [score_checklist(check, True, False) for check in per_rater]
        for field in BINARY_FIELDS:
            rater_flags[field].append([row[field] for row in scored])
        # 체크 항목별 다수결. 동률이면 보수적으로(문제 있음, 부분 포함) 정하고 논의 목록에 남긴다.
        points = []
        for i in range(len(per_rater[0]["required_points"])):
            status, tied = majority([c["required_points"][i]["status"] for c in per_rater],
                                    "partial")
            points.append({"index": i + 1, "status": status})
            if tied:
                ties.append(f"{item_id} rp{i + 1}")
        merged = {"required_points": points, "forbidden_claims": [], "unsupported_claims": []}
        hallucination, tied = majority([row["hallucination"] for row in scored], True)
        if tied:
            ties.append(f"{item_id} hallucination")
        if hallucination:
            merged["unsupported_claims"] = ["(다수결: 환각 있음)"]
        for field in FLAG_COLUMNS:
            merged[field], tied = majority([c[field] for c in per_rater], True)
            if tied:
                ties.append(f"{item_id} {field}")
        human = score_checklist(merged, True, foreign_auto(record))
        llm = (judgements.get(experiment_id) or {}).get("metrics")
        items.append({"item_id": item_id, "experiment_id": experiment_id,
                      "alias": record["model_info"]["alias"], "human": human, "llm": llm})

    if problems:
        print("\n".join(problems))
    agreement = {"raters": sorted(sheets), "items": len(items), "ties": ties,
                 "problems": problems,
                 "fleiss_kappa": {field: fleiss_kappa(rater_flags[field])
                                  for field in BINARY_FIELDS},
                 "llm_vs_human": {}}
    paired = [item for item in items if item["llm"]]
    for field in (*BINARY_FIELDS, "usable"):
        humans = [item["human"][field] for item in paired]
        llms = [item["llm"][field] for item in paired]
        agreement["llm_vs_human"][field] = {
            "agreement": sum(h == m for h, m in zip(humans, llms, strict=True)) / len(paired)
            if paired else None,
            "cohen_kappa": cohen_kappa(humans, llms), "pabak": pabak(humans, llms)}
    # 평가 기준 7장: 이 판정으로 LLM 채점(전체)과 사람 다수결(36건) 중 무엇을 우선할지 정한다.
    agreement["llm_trust_decision"] = llm_trust_decision(
        [item["human"]["usable"] for item in paired], [item["llm"]["usable"] for item in paired])
    agreement["llm_vs_human"]["deduction_score_pearson"] = pearson(
        [item["human"]["deduction_score"] for item in paired],
        [item["llm"]["deduction_score"] for item in paired])
    by_alias = {}
    for item in items:
        by_alias.setdefault(item["alias"], []).append(item["human"])
    write_json(directory / ("quality_human_dummy.json" if dummy else "quality_human.json"),
               {"is_dummy": dummy, "sheet_dir": sheet_dir.name,
                "items": items, "agreement": agreement,
                "targets": {alias: summarize_scores(rows) for alias, rows in by_alias.items()},
                "updated_at": now_iso()})
    print(json.dumps(agreement, ensure_ascii=False, indent=2))
    return agreement


def select_generation(directory, dummy=False):
    """평가 기준 7·8장대로 품질 지표의 출처를 정하고 서비스에 쓸 생성 모델을 고른다.

    status: final(실제 사람 채점 반영) / provisional(사람 채점 전, LLM 채점만) / dummy(가상 채점 미리보기)
    """
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    llm = json.loads((directory / "quality_llm.json").read_text(encoding="utf-8"))
    human_path = directory / ("quality_human_dummy.json" if dummy else "quality_human.json")
    human = json.loads(human_path.read_text(encoding="utf-8")) if human_path.exists() else None
    if human is None:
        status, source, trust = "provisional", "llm", None
    else:
        trust = human["agreement"]["llm_trust_decision"]
        source = "llm" if trust["use_llm_scores"] else "human"
        status = "dummy" if human.get("is_dummy") else "final"
    targets = llm["targets"] if source == "llm" else human["targets"]
    decision = choose_generation_model(
        targets, {alias: row["model_type"] for alias, row in summary["targets"].items()},
        {alias: row["latency_mean_sec"] for alias, row in summary["targets"].items()})
    result = {"status": status, "quality_source": source, "llm_trust_decision": trust,
              "comparison": directory.name, "prompt_version": summary["prompt_version"],
              "structured_output": summary.get("structured_output"), **decision,
              "targets": targets, "operations": summary["targets"], "selected_at": now_iso()}
    name = "generation_selected_dummy.json" if dummy else "generation_selected.json"
    write_json(directory / name, result)
    lesson.SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    write_json(lesson.SELECTION_DIR / name, result)
    print(f"[{status}] 품질 출처 {source} → 선택: {decision['selected']} ({decision['reason']})")
    for alias, check in decision["local_checks"].items():
        print(f"  {alias}: 사용 가능률 차이 {check['usable_gap']:+.3f}, "
              f"환각률 차이 {check['hallucination_gap']:+.3f}, 통과 {check['passes']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="생성 요약 품질 채점")
    parser.add_argument("action", choices=["judge", "sheet", "aggregate", "select"])
    parser.add_argument("--dir", type=Path, help="비교 폴더. 생략하면 가장 최근 폴더")
    parser.add_argument("--dummy", action="store_true",
                        help="aggregate: human_dummy/의 가상 채점 시트를 집계(미리보기용)")
    args = parser.parse_args()
    directory = args.dir or latest_comparison()
    print("비교 폴더:", directory)
    if args.action == "judge":
        print(json.dumps(run_judge(directory), ensure_ascii=False, indent=2))
    elif args.action == "sheet":
        make_sheets(directory)
    elif args.action == "aggregate":
        aggregate(directory, dummy=args.dummy)
    else:
        select_generation(directory, dummy=args.dummy)


if __name__ == "__main__":
    main()
