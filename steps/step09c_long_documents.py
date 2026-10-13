"""09c. 긴 문서의 생성 시간·토큰·비용을 원문 그대로와 본문만 추출한 입력으로 비교한다.

입력은 심화 validation 60건(참고 자료가 섞인 긴 문서 포함). 채점 기준이 없으므로 품질은 채점하지 않고,
시간·토큰·비용·형식 성공률과 분류 정확도(validation 라벨)를 기록한다.
서비스와 같은 흐름으로 조건마다 분류부터 다시 한다.
- raw: 원문 그대로 분류하고 원문 그대로 요약
- body: common.preprocess.extract_body로 참고 자료를 뗀 본문을 분류하고 요약

참고 자료 유입 의심도: 요약문 글자 4-그램 중 참고 자료에만 있는 비율(대략적인 지표, 사람이 확인할 사례 고르기용).
openai를 넣으면 요금이 발생한다. 심화 test는 읽지 않는다.

uv run python steps/step09c_long_documents.py --targets qwen2.5-7b qwen3-4b openai
"""

import argparse
import json

import lesson_settings as lesson
from common.preprocess import REFERENCE_END, REFERENCE_HEADER, document_layout, extract_body
from common.records import new_run_dir, now_iso, write_json
from prompts.document_policies import PROMPT_VERSION
from step01_read_data import read_rows
from step06_predict import load_classifier
from step07_call_models import execution_conditions, ollama_unload, resolve_target
from step08_connect_chain import create_chain, generate_prepared
from step09_compare import _mean, _percentile, build_record, warm_up

DATASET = "documents_advanced"
DATA = lesson.ROOT / "data" / DATASET
CONDITIONS = {"raw": lambda text: text, "body": extract_body}


def reference_part(text):
    if REFERENCE_HEADER not in text or REFERENCE_END not in text:
        return ""
    return text[text.index(REFERENCE_HEADER):text.index(REFERENCE_END)]


def grams(text, n=4):
    text = "".join(text.split())
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def reference_leak(summary, original):
    """요약문 4-그램 중 참고 자료에만 있고 본문에는 없는 비율."""
    reference = reference_part(original)
    if not reference or not summary:
        return None
    own = grams(summary)
    only_reference = grams(reference) - grams(extract_body(original))
    return len(own & only_reference) / len(own) if own else None


def summary_text(parsed):
    if not isinstance(parsed, dict):
        return ""
    values = []
    for value in parsed.values():
        values.extend(value if isinstance(value, list) else [value])
    return " ".join(str(value) for value in values if value)


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault((record["model_info"]["alias"], record["condition"]), []).append(record)
    table = {}
    for (alias, condition), rows in groups.items():
        success = [row for row in rows if row["result"]["status"] == "SUCCESS"]

        def metric(key, source=success):
            return [row["performance_metrics"][key] for row in source]

        costs = [row["cost_metrics"]["api_cost_usd"] for row in rows
                 if row["cost_metrics"]["api_cost_usd"] is not None]
        long_rows = [row for row in success if row["layout"] != "short"]
        leaks = [row["reference_leak"] for row in rows if row["reference_leak"] is not None]
        table[f"{alias}|{condition}"] = {
            "alias": alias, "condition": condition, "calls": len(rows),
            "errors": len(rows) - len(success),
            "format_valid_strict_rate": sum(r["result"]["format_valid_strict"] for r in rows)
            / len(rows),
            "classification_accuracy": sum(r["classification_correct"] for r in rows) / len(rows),
            "input_chars_mean": _mean([row["input_chars"] for row in rows]),
            "prompt_tokens_mean": _mean(metric("prompt_tokens")),
            "prompt_tokens_mean_long_only": _mean(metric("prompt_tokens", long_rows)),
            "completion_tokens_mean": _mean(metric("completion_tokens")),
            "latency_mean_sec": _mean(metric("total_time_sec")),
            "latency_p95_sec": _percentile(metric("total_time_sec"), 0.95),
            "latency_mean_sec_long_only": _mean(metric("total_time_sec", long_rows)),
            "ttft_mean_sec": _mean(metric("ttft_sec")),
            "api_cost_total_usd": sum(costs) if costs else None,
            "api_cost_per_1000_docs_usd": sum(costs) / len(rows) * 1000 if costs else None,
            "reference_leak_mean": _mean(leaks),
            "reference_leak_over_0_2": sum(leak > 0.2 for leak in leaks),
            "reference_leak_rows": len(leaks)}
    return table


def main():
    parser = argparse.ArgumentParser(description="긴 문서 시간·비용 비교(원문 vs 본문)")
    parser.add_argument("--targets", nargs="+", default=["qwen2.5-7b", "qwen3-4b"])
    parser.add_argument("--limit", type=int, default=0, help="앞에서 몇 건만(시험용). 0이면 전부")
    args = parser.parse_args()
    rows = read_rows(DATA / "validation.csv")
    rows = rows[:args.limit] if args.limit else rows
    classifier = load_classifier()
    chain = create_chain(classifier, extract=False)  # 조건별 입력을 이 스크립트가 직접 정한다
    structured = lesson.STRUCTURED_OUTPUT
    targets = [resolve_target(name) for name in args.targets]
    output_dir = new_run_dir(lesson.OUTPUT_ROOT / DATASET / lesson.LEARNER, "long_documents",
                             subdir="comparisons")
    prepared = {condition: [chain.invoke(prepare(row["text"])) for row in rows]
                for condition, prepare in CONDITIONS.items()}
    conditions = {t["alias"]: execution_conditions(t, structured) for t in targets}
    write_json(output_dir / "config.json", {
        "purpose": "긴 문서 생성 시간·토큰·비용: 원문 vs 본문만", "dataset": DATASET,
        "split": "validation", "rows": len(rows), "prompt_version": PROMPT_VERSION,
        "structured_output": structured, "classifier_run": classifier["identity"]["run_name"],
        "conditions": {"raw": "원문 그대로 분류·요약", "body": "참고 자료를 뗀 본문으로 분류·요약"},
        "quality_scored": False, "targets": targets, "execution_conditions": conditions,
        "started_at": now_iso()})

    records, warmups = [], {}
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for target in targets:
            warmups[target["alias"]] = warm_up(target, structured)
            print(f"\n[{target['alias']}] 준비 호출 {warmups[target['alias']]['status']}")
            for condition in CONDITIONS:
                for row, fixed in zip(rows, prepared[condition], strict=True):
                    generated = generate_prepared(fixed, target, structured)
                    case = {"id": row["id"], "expected_label": row["label"]}
                    record = build_record(output_dir.name, case, fixed, target,
                                          conditions[target["alias"]], generated, 1)
                    record["experiment_id"] += f"-{condition}"
                    record.update({
                        "condition": condition, "layout": document_layout(row["text"]),
                        "input_chars": len(fixed["text"]),
                        "classification_correct": fixed["classification"]["label"] == row["label"],
                        "reference_leak": reference_leak(summary_text(generated["parsed"]),
                                                         row["text"])})
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    records.append(record)
                    print(f"  {condition} {row['id']}: {generated['status']} "
                          f"{generated['performance_metrics']['total_time_sec']:.1f}s "
                          f"입력 {len(fixed['text'])}자")
            if target["provider"] == "ollama":
                ollama_unload(target["model"])
    write_json(output_dir / "warmup.json", warmups)
    summary = {"rows": len(rows), "prompt_version": PROMPT_VERSION, "quality_scored": False,
               "table": summarize(records), "finished_at": now_iso()}
    write_json(output_dir / "summary.json", summary)
    for key, value in summary["table"].items():
        print(key, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in value.items()}, ensure_ascii=False))
    print("저장 위치:", output_dir)


if __name__ == "__main__":
    main()
