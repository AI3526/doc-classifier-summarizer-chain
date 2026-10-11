"""09. 같은 입력과 프롬프트로 여러 생성 모델을 호출하고 비교 기록을 저장한다.

입력 묶음(--split)
- smoke: steps/examples의 2건. 코드가 도는지만 확인한다.
- development: generation_development.jsonl 15건. 1차 비교와 프롬프트 개발에 쓴다.
- eval: generation_eval.jsonl 24건. 최종 비교. 이 답변을 보고 프롬프트를 고치지 않는다.

모델마다 준비 호출로 먼저 메모리에 올린 뒤 측정하고, 끝나면 내린다(로컬 모델).
분류는 입력마다 한 번만 하고, 같은 프롬프트를 모든 모델이 공유한다.
expected_label, required_points 등은 채점용으로만 기록하고 프롬프트에 넣지 않는다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step09_compare.py --split smoke
uv run python steps/step09_compare.py --split development
uv run python steps/step09_compare.py --split eval --targets qwen2.5-7b exaone3.5-7.8b openai --repeats 2
uv run python steps/step09_compare.py --split development --structured   # 구조화 출력
"""

import argparse
import json
import statistics
from pathlib import Path

import lesson_settings as lesson
from common.records import new_run_dir, now_iso, write_json
from prompts.document_policies import PROMPT_VERSION, build_prompt, response_schema
from step01_read_data import read_jsonl, read_rows
from step06_predict import load_classifier
from step07_call_models import call_model, execution_conditions, ollama_unload, resolve_target
from step08_connect_chain import MAX_INPUT_CHARS, create_chain, generate_prepared

SPLITS = {
    "smoke": lesson.DEVELOPMENT_INPUT,
    "development": lesson.DATA_DIR / "generation_development.jsonl",
    "eval": lesson.DATA_DIR / "generation_eval.jsonl",
}
# 비용이 드는 openai는 기본 목록에서 빼고, 필요할 때 --targets로 직접 넣는다.
DEFAULT_TARGETS = [name for name, spec in lesson.GENERATION_CANDIDATES.items()
                   if spec["provider"] == "ollama"]
SCORING_FIELDS = ("expected_label", "required_points", "forbidden_claims", "rubric_note")


def read_cases(path):
    cases = read_jsonl(path)
    validate_cases(cases)
    return cases


def validate_cases(cases):
    # 빈 입력이나 중복 id를 발견하면 유료 호출 전에 멈춘다.
    if not cases:
        raise ValueError("비교 입력이 비어 있습니다.")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)) or any(not str(value).strip() for value in ids):
        raise ValueError("입력 id는 내용이 있어야 하고 서로 달라야 합니다.")
    for case in cases:
        text = case["text"]
        if not text.strip() or len(text) > MAX_INPUT_CHARS:
            raise ValueError("비교 입력의 text와 길이를 확인하세요.")


def warm_up(target, structured=False):
    """측정 전에 모델을 메모리에 올린다.

    비교 입력과 겹치지 않게 train 첫 문장을 쓰고, 결과는 warmup.json에만 남기며 요약 지표에서 뺀다.
    적재 시간(콜드 스타트)은 운영 조건 비교용으로 따로 보고한다.
    """
    row = read_rows(lesson.DATA_DIR / "train.csv")[0]
    result = call_model(target, build_prompt(row["text"], row["label"]),
                        response_format=response_schema(row["label"]) if structured else None)
    return {"alias": target["alias"], "sample_id": row["id"], "excluded_from_metrics": True,
            "status": result["status"], "error": result["error"],
            "error_message": result["error_message"],
            "load_time_sec": result["performance_metrics"]["load_time_sec"],
            "total_time_sec": result["performance_metrics"]["total_time_sec"]}


def build_record(run_name, case, prepared, target, conditions, generated, repeat):
    return {
        "experiment_id": f"{run_name}-{target['alias']}-{case['id']}-r{repeat}",
        "task_type": "document_summary", "case_id": case["id"], "repeat": repeat,
        "input_data": {
            "sample_id": case["id"],
            "predicted_label": prepared["classification"]["label"],
            "classification_confidence": prepared["classification"]["confidence"],
            "prompt_template_used": prepared["prompt_version"],
            "source_text": prepared["text"],
        },
        # 채점 전용. 프롬프트에는 들어가지 않았다.
        "scoring_reference": {key: case.get(key) for key in SCORING_FIELDS},
        "model_info": {"model_type": "local" if target["provider"] == "ollama" else "cloud",
                       "model_name": target["model"], "provider": target["provider"],
                       "alias": target["alias"]},
        "execution_conditions": conditions,
        "performance_metrics": generated["performance_metrics"],
        "cost_metrics": generated["cost_metrics"],
        "result": {
            "status": generated["status"], "error": generated["error"],
            "error_message": generated["error_message"],
            "generated_text": generated["text"],
            "generation_complete": generated["generation_complete"],
            "finish_reason": generated["finish_reason"],
            "format_valid_strict": generated["format_valid_strict"],
            "format_valid_lenient": generated["format_valid_lenient"],
            "format_error": generated["format_error"], "parsed": generated["parsed"],
        },
        "prompt": prepared["prompt"],
        "timestamp": now_iso(),
    }


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def _percentile(values, ratio):
    values = sorted(value for value in values if value is not None)
    return values[min(len(values) - 1, int(len(values) * ratio))] if values else None


def summarize_target(records, alias, warmup):
    subset = [row for row in records if row["model_info"]["alias"] == alias]
    success = [row for row in subset if row["result"]["status"] == "SUCCESS"]
    calls = len(subset)

    def metric(key):
        return [row["performance_metrics"][key] for row in success]

    costs = [row["cost_metrics"]["api_cost_usd"] for row in subset]
    known_costs = [cost for cost in costs if cost is not None]
    mean_cost = statistics.mean(known_costs) if known_costs else None
    return {
        "model": subset[0]["model_info"]["model_name"] if subset else None,
        "model_type": subset[0]["model_info"]["model_type"] if subset else None,
        "calls": calls, "errors": calls - len(success),
        "error_rate": (calls - len(success)) / calls if calls else None,
        "format_valid_strict_rate": sum(r["result"]["format_valid_strict"] for r in subset) / calls
        if calls else None,
        "format_valid_lenient_rate": sum(r["result"]["format_valid_lenient"] for r in subset)
        / calls if calls else None,
        "generation_complete_rate": sum(r["result"]["generation_complete"] for r in subset)
        / calls if calls else None,
        "latency_mean_sec": _mean(metric("total_time_sec")),
        "latency_p50_sec": _percentile(metric("total_time_sec"), 0.5),
        "latency_p95_sec": _percentile(metric("total_time_sec"), 0.95),
        "ttft_mean_sec": _mean(metric("ttft_sec")),
        "tokens_per_second_mean": _mean(metric("tokens_per_second")),
        "prompt_tokens_mean": _mean(metric("prompt_tokens")),
        "completion_tokens_mean": _mean(metric("completion_tokens")),
        "vram_mb_max": max((v for v in metric("vram_mb") if v is not None), default=None),
        "memory_mb_max": max((v for v in metric("peak_memory_mb") if v is not None),
                             default=None),
        "warmup_load_time_sec": warmup.get("load_time_sec"),
        "api_cost_total_usd": sum(known_costs) if known_costs else None,
        "api_cost_per_1000_docs_usd": mean_cost * 1000 if mean_cost is not None else None,
        "energy_wh_total_estimate": sum(e for e in (r["cost_metrics"]["energy_wh_estimate"]
                                                    for r in subset) if e is not None) or None,
    }


def write_summary_md(path, summary):
    columns = [("calls", "호출"), ("errors", "오류"), ("format_valid_strict_rate", "형식(엄격)"),
               ("format_valid_lenient_rate", "형식(완화)"), ("latency_mean_sec", "응답 평균(s)"),
               ("latency_p95_sec", "응답 p95(s)"), ("ttft_mean_sec", "TTFT(s)"),
               ("tokens_per_second_mean", "토큰/s"), ("completion_tokens_mean", "출력 토큰"),
               ("vram_mb_max", "VRAM(MB)"), ("warmup_load_time_sec", "적재(s)"),
               ("api_cost_per_1000_docs_usd", "1000건 비용($)")]

    def cell(value):
        if value is None:
            return "-"
        return f"{value:.3f}" if isinstance(value, float) and value < 10 else (
            f"{value:.1f}" if isinstance(value, float) else str(value))

    lines = [f"# 생성 모델 비교 ({summary['split']}, {summary['cases']}건 × {summary['repeats']}회)",
             "", (f"- 프롬프트: `{summary['prompt_version']}`, "
                  f"구조화 출력: {'사용' if summary.get('structured_output') else '미사용'}, "
                  f"분류기: `{summary['classifier_run']}`"),
             (f"- 분류 오답 입력: {len(summary['misclassified_cases'])}건 "
              f"{summary['misclassified_cases'] or ''}"), "",
             "| 모델 | " + " | ".join(name for _, name in columns) + " |",
             "|---|" + "---|" * len(columns)]
    for alias, row in summary["targets"].items():
        lines.append(f"| {alias} | " + " | ".join(cell(row[key]) for key, _ in columns) + " |")
    lines += ["", "품질 점수는 채점 단계에서 따로 계산한다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_comparison(chain, cases, targets, output_dir, split, repeats=1, classifier_run=None,
                   structured=False):
    validate_cases(cases)
    if repeats < 1:
        raise ValueError("--repeats는 1 이상이어야 합니다.")
    targets = [resolve_target(name) for name in targets]
    prepared = [chain.invoke(case["text"]) for case in cases]  # text만 넘긴다.
    conditions = {target["alias"]: execution_conditions(target, structured) for target in targets}
    write_json(output_dir / "config.json", {
        "split": split, "cases": len(cases), "repeats": repeats, "prompt_version": PROMPT_VERSION,
        "structured_output": structured,
        "classifier_run": classifier_run, "targets": targets,
        "execution_conditions": conditions, "started_at": now_iso()})

    records, warmups = [], {}
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        # 모델을 바깥 반복에 두어 모델마다 한 번만 메모리에 올린다.
        for target in targets:
            warmups[target["alias"]] = warm_up(target, structured)
            print(f"\n[{target['alias']}] 준비 호출 {warmups[target['alias']]['status']}, "
                  f"적재 {warmups[target['alias']]['load_time_sec']}")
            for repeat in range(1, repeats + 1):
                for case, fixed in zip(cases, prepared, strict=True):
                    generated = generate_prepared(fixed, target, structured)
                    record = build_record(output_dir.name, case, fixed, target,
                                          conditions[target["alias"]], generated, repeat)
                    # 실패한 호출도 저장한다. 중단되더라도 끝난 호출 기록은 남긴다.
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    records.append(record)
                    print(f"  r{repeat} {case['id']}: {generated['status']} "
                          f"{generated['performance_metrics']['total_time_sec']:.1f}s "
                          f"형식 {'O' if generated['format_valid_strict'] else 'X'}")
            if target["provider"] == "ollama":
                ollama_unload(target["model"])
    write_json(output_dir / "warmup.json", warmups)

    misclassified = sorted({row["case_id"] for row in records
                            if row["scoring_reference"]["expected_label"]
                            and row["input_data"]["predicted_label"]
                            != row["scoring_reference"]["expected_label"]})
    summary = {"split": split, "cases": len(cases), "repeats": repeats,
               "prompt_version": PROMPT_VERSION, "structured_output": structured,
               "classifier_run": classifier_run, "misclassified_cases": misclassified, "quality_scored": False,
               "targets": {target["alias"]: summarize_target(records, target["alias"],
                                                             warmups[target["alias"]])
                           for target in targets},
               "finished_at": now_iso()}
    write_json(output_dir / "summary.json", summary)
    write_summary_md(output_dir / "summary.md", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="생성 모델 비교")
    parser.add_argument("--split", choices=list(SPLITS), default="smoke")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                        help=f"후보: {', '.join(lesson.GENERATION_CANDIDATES)}")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--name", default="", help="결과 폴더 이름에 붙일 메모")
    parser.add_argument("--structured", action="store_true",
                        help="출력 스키마로 구조화 출력을 요청한다(Ollama format, OpenAI json_schema)")
    args = parser.parse_args()
    cases = read_cases(SPLITS[args.split])
    classifier = load_classifier()
    # artifacts/{dataset}/{learner}/comparisons/{시각}_{split}_{name}
    output_dir = new_run_dir(lesson.LEARNER_DIR, "_".join(filter(None, [args.split, args.name])),
                             subdir="comparisons")
    summary = run_comparison(create_chain(classifier), cases, args.targets, output_dir,
                             args.split, args.repeats, classifier["identity"]["run_name"],
                             args.structured)
    print(json.dumps(summary["targets"], ensure_ascii=False, indent=2))
    print("저장 위치:", output_dir)


if __name__ == "__main__":
    main()
