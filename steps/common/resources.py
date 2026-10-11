"""시간과 메모리 측정. 측정 범위를 결과에 함께 남긴다."""

import importlib.metadata
import platform
import statistics
import time

import psutil
import torch


def measure_latency(predict_one, texts, warmup=3):
    """문장 한 개씩 예측하는 시간을 잰다. 처음 몇 번은 준비 시간이라 제외한다."""
    for text in texts[:warmup]:
        predict_one(text)
    times = []
    for text in texts:
        started = time.perf_counter()
        predict_one(text)
        times.append((time.perf_counter() - started) * 1000)
    ordered = sorted(times)
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "samples": len(times),
        "warmup_calls": min(warmup, len(texts)),
        "scope": "1건씩 전처리부터 예측 라벨까지, 모델 로딩 제외",
    }


def reset_gpu_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb():
    # allocated는 텐서가 실제로 쓴 최대치, reserved는 PyTorch가 잡아 둔 최대치다.
    if not torch.cuda.is_available():
        return {"peak_vram_allocated_mb": None, "peak_vram_reserved_mb": None}
    return {
        "peak_vram_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_vram_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
    }


def process_rss_mb():
    return psutil.Process().memory_info().rss / 2**20


def environment_info():
    packages = {}
    for name in ("torch", "transformers", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "cpu_threads_torch": torch.get_num_threads(),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": packages,
    }
