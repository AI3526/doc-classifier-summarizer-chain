"""서비스에서 분류기와 로컬 생성 LLM을 함께 실행할 때의 메모리와 처리 시간을 잰다.

순서: 로컬 모델을 모두 내린 상태(기준) → 분류기 로딩(CPU) → 로컬 LLM 적재(GPU) → 문서 몇 건을
서비스 체인(분류 → 정책 → 생성)으로 처리. 단계마다 이 프로세스의 RAM, 시스템 RAM, GPU 메모리를 기록한다.
입력은 generation_development.jsonl의 text만 쓴다. 로컬 모델만 쓰므로 요금은 없다.

uv run python scripts/measure_service_memory.py
uv run python scripts/measure_service_memory.py --targets qwen2.5-7b qwen3-4b
결과: artifacts/{dataset}/service/memory.json, reports/generated/service_memory.md
"""

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

import lesson_settings as lesson
import psutil
from common.records import now_iso, write_json
from step01_read_data import read_jsonl
from step06_predict import load_classifier, predict_text
from step07_call_models import ollama_memory_mb, ollama_unload, resolve_target
from step08_connect_chain import create_service_chain

OUTPUT = lesson.OUTPUT_ROOT / lesson.DATASET / "service" / "memory.json"
REPORT = ROOT / "reports" / "generated" / "service_memory.md"
DOCS = 6


def gpu_used_mb():
    """nvidia-smi로 GPU 전체 사용 메모리(MB). 다른 프로그램의 사용량도 포함된다."""
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10, check=True).stdout
        used, total = (float(value) for value in output.strip().splitlines()[0].split(","))
        return used, total
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, None


def snapshot(stage):
    process = psutil.Process()
    memory = psutil.virtual_memory()
    gpu_used, gpu_total = gpu_used_mb()
    return {"stage": stage, "process_rss_mb": process.memory_info().rss / 2**20,
            "system_ram_used_mb": (memory.total - memory.available) / 2**20,
            "system_ram_total_mb": memory.total / 2**20,
            "gpu_used_mb": gpu_used, "gpu_total_mb": gpu_total}


def measure_target(classifier, target, texts):
    spec = resolve_target(target)
    ollama_unload(spec["model"])
    chain = create_service_chain(classifier, target)
    stages = [snapshot("before_load")]
    timings = []
    for index, text in enumerate(texts):
        started = time.perf_counter()
        result = chain.invoke(text)
        elapsed = time.perf_counter() - started
        generation = result["generation"]
        if index == 0:
            # 첫 요청에는 모델 적재(콜드 스타트)가 들어 있어 따로 기록한다.
            total_mb, vram_mb = ollama_memory_mb(spec["model"])
            stages.append({**snapshot("after_first_request"), "ollama_model_mb": total_mb,
                           "ollama_vram_mb": vram_mb, "first_request_sec": elapsed,
                           "load_time_sec": generation["performance_metrics"]["load_time_sec"]})
        else:
            timings.append({"end_to_end_sec": elapsed,
                            "generation_sec": generation["performance_metrics"]["total_time_sec"],
                            "status": generation["status"]})
    stages.append(snapshot("after_requests"))
    ollama_unload(spec["model"])
    return {"target": target, "model": spec["model"], "stages": stages, "requests": timings,
            "end_to_end_mean_sec": statistics.mean(t["end_to_end_sec"] for t in timings),
            "classification_overhead_mean_sec": statistics.mean(
                t["end_to_end_sec"] - t["generation_sec"] for t in timings)}


def main():
    parser = argparse.ArgumentParser(description="분류기 + 로컬 LLM 동시 실행 메모리 측정")
    parser.add_argument("--targets", nargs="+", default=["qwen2.5-7b", "qwen3-4b"])
    args = parser.parse_args()
    for spec in lesson.GENERATION_CANDIDATES.values():
        if spec["provider"] == "ollama":
            ollama_unload(spec["model"])
    time.sleep(2)  # 언로드가 GPU 메모리에 반영될 시간을 준다.
    baseline = snapshot("baseline (로컬 모델 없음, 분류기 전)")
    classifier = load_classifier()
    # 가중치는 메모리 매핑으로 읽혀 로딩 직후에는 RAM에 거의 잡히지 않는다.
    # 한 번 분류해 가중치를 실제로 읽은 뒤 잰다.
    predict_text("메모리 측정을 위한 준비 분류 문장입니다.", classifier)
    loaded = snapshot("classifier_loaded_and_used")
    texts = [case["text"] for case in read_jsonl(lesson.DATA_DIR / "generation_development.jsonl")
             ][:DOCS]
    results = [measure_target(classifier, target, texts) for target in args.targets]
    report = {"measured_at": now_iso(), "classifier": classifier["identity"],
              "classifier_device": "cpu", "baseline": baseline, "classifier_loaded": loaded,
              "classifier_rss_mb": loaded["process_rss_mb"] - baseline["process_rss_mb"],
              "documents": len(texts), "targets": results}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT, report)

    def f(value, digits=0):
        return "-" if value is None else f"{value:,.{digits}f}"

    lines = ["# 분류기 + 로컬 LLM 동시 실행 메모리", "",
             (f"`scripts/measure_service_memory.py`가 생성. 측정 시각 {report['measured_at']}. "
             f"분류기는 CPU, 로컬 LLM은 Ollama(GPU)에서 실행."), "",
             (f"- 분류기 로딩으로 늘어난 프로세스 RAM: **{f(report['classifier_rss_mb'])}MB** "
             f"({classifier['identity']['run_name']})"),
             (f"- 기준 GPU 사용량(로컬 모델 없음, 다른 프로그램 포함): {f(baseline['gpu_used_mb'])}MB / "
             f"{f(baseline['gpu_total_mb'])}MB"), "",
             ("| 생성 모델 | Ollama 모델 메모리 | 그중 VRAM | GPU 사용량(함께 실행) | 시스템 RAM 사용량 | "
             "콜드 스타트 첫 요청 | 요청 평균(분류+생성) | 분류·정책 오버헤드 |"),
             "|---|---|---|---|---|---|---|---|"]
    for row in results:
        first = row["stages"][1]
        lines.append(f"| {row['target']} | {f(first['ollama_model_mb'])}MB | {f(first['ollama_vram_mb'])}MB"
                     f" | {f(first['gpu_used_mb'])}MB | {f(first['system_ram_used_mb'])}MB | "
                     f"{first['first_request_sec']:.2f}s | {row['end_to_end_mean_sec']:.2f}s | "
                     f"{row['classification_overhead_mean_sec'] * 1000:.0f}ms |")
    lines += ["", (f"요청 평균은 첫 요청을 뺀 {len(texts) - 1}건 기준이다. GPU·시스템 RAM 사용량에는 "
              "측정 시점에 실행 중인 다른 프로그램도 포함된다.")]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
