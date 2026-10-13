"""로컬 GPU 서버와 상용 API의 손익분기 문서 수를 계산한다.

상용 API 비용과 로컬 응답 시간은 실측(생성 비교 기록)에서 가져오고, 서버 장비값·전력·유지보수는
아래 가정값(추정)을 쓴다. 가정값은 출처와 확인 날짜를 함께 남긴다.

- 짧은 문서: documents 최종 비교(eval v6)의 실측
- 긴 문서 원문 / 본문만: documents_advanced 긴 문서 측정(step09c)의 실측

uv run python scripts/break_even.py
결과: artifacts/documents_advanced/break_even/break_even.json, reports/generated/break_even.md
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

import lesson_settings as lesson
from common.records import now_iso, write_json

SHORT_SUMMARY = (lesson.OUTPUT_ROOT / "documents" / lesson.LEARNER / "comparisons"
                 / "20261012-164245_eval_final_v6" / "summary.json")
LONG_DIR = lesson.OUTPUT_ROOT / "documents_advanced" / lesson.LEARNER / "comparisons"
OUTPUT = lesson.OUTPUT_ROOT / "documents_advanced" / "break_even"
REPORT = ROOT / "reports" / "generated" / "break_even.md"
LOCAL_TARGET = "qwen2.5-7b"   # 로컬 기본 후보(기밀 문서 라우팅 대상)
CLOUD_TARGET = "openai"

# ---- 가정값 (추정). 2026-10-13 검색으로 확인한 범위에서 대표값을 고름 ----
KRW_PER_USD = 1400            # 환율 가정
ELECTRICITY_KRW_PER_KWH = 182  # 산업용 평균 판매단가 수준(2025년 181.9원/kWh 보도)
DEPRECIATION_MONTHS = 36      # 3년 감가상각
MAINTENANCE_RATE_PER_YEAR = 0.10  # 장비값의 연 10% (보증·부품·교체)
OPS_LABOR_USD_PER_MONTH = 1000    # 관리 인력 일부(추정). 회사마다 크게 달라 별도 시나리오로만 씀
CLOUD_PRICE_MULTIPLIERS = (1, 10)  # 상용 단가가 지금의 10배인 더 비싼 모델일 때도 함께 계산
SERVERS = {
    "small": {"name": "RTX 5090 32GB 1장 워크스테이션형 서버", "price_usd": 8000, "power_kw": 0.8,
              "basis": "완성 워크스테이션 약 $6,000~9,500 (카드 품귀로 출시가보다 높음)"},
    "medium": {"name": "L40S 48GB 1장 서버", "price_usd": 15000, "power_kw": 0.6,
               "basis": "카드 약 $7,500, 4장 서버(Dell R760xa) 약 $47,000~49,000 → 1장 서버 추정"},
    "large": {"name": "H100 80GB PCIe 1장 서버", "price_usd": 40000, "power_kw": 0.7,
              "basis": "카드 약 $22,000~32,000 + 서버 본체 추정. 7B 모델에는 과한 구성"},
}
SOURCES = [
    "https://modal.com/blog/nvidia-l40s-price-article",
    "https://www.hardwaredealz.com/hardware/gpu/nvidia-l40s-48gb/nvidia-48gb-gddr6-4x-dp/preisvergleich.2493",
    "https://techradar.com/pro/rtx-5090s-are-going-for-as-much-as-usd9000-as-ai-server-builders-reportedly-buy-nvidias-flagship-by-the-pallet-but-the-photos-behind-the-story-may-be-ai-generated",
    "https://book.petronellatech.com/blog/how-to-build-custom-ai-workstation-2026/",
    "https://www.morphllm.com/nvidia-h100",
    "https://gpuperhour.com/blog/nvidia-h100-price",
    "https://www.hankyung.com/article/202603238972Y",
    "https://blog.haezoom.com/notice_30/",
]
HOURS_PER_MONTH = 24 * 30


def latest_long_summary():
    folders = sorted(LONG_DIR.glob("*_long_documents/summary.json"))
    if not folders:
        raise FileNotFoundError("먼저 steps/step09c_long_documents.py를 실행하세요.")
    return folders[-1]


def scenarios():
    short = json.loads(SHORT_SUMMARY.read_text(encoding="utf-8"))["targets"]
    long_path = latest_long_summary()
    table = json.loads(long_path.read_text(encoding="utf-8"))["table"]
    rows = {"short": {"label": "짧은 문서(기본 데이터), 평균 약 100자",
                      "cloud_cost_per_doc": short[CLOUD_TARGET]["api_cost_per_1000_docs_usd"] / 1000,
                      "local_latency_sec": short[LOCAL_TARGET]["latency_mean_sec"],
                      "prompt_tokens": short[CLOUD_TARGET]["prompt_tokens_mean"]}}
    for condition, label in (("raw", "긴 문서 원문 (참고 자료 포함)"), ("body", "긴 문서 본문만")):
        cloud, local = table[f"{CLOUD_TARGET}|{condition}"], table[f"{LOCAL_TARGET}|{condition}"]
        rows[condition] = {"label": f"{label}, 평균 {cloud['input_chars_mean']:.0f}자",
                           "cloud_cost_per_doc": cloud["api_cost_per_1000_docs_usd"] / 1000,
                           "local_latency_sec": local["latency_mean_sec"],
                           "prompt_tokens": cloud["prompt_tokens_mean"]}
    return rows, [SHORT_SUMMARY, long_path]


def monthly_cost(server, with_labor):
    power = server["power_kw"] * HOURS_PER_MONTH * ELECTRICITY_KRW_PER_KWH / KRW_PER_USD
    cost = (server["price_usd"] / DEPRECIATION_MONTHS
            + server["price_usd"] * MAINTENANCE_RATE_PER_YEAR / 12 + power)
    return cost + (OPS_LABOR_USD_PER_MONTH if with_labor else 0), power


def main():
    rows, inputs = scenarios()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for key, server in SERVERS.items():
        for with_labor in (False, True):
            monthly, power = monthly_cost(server, with_labor)
            for scenario, data in rows.items():
                # 노트북 GPU(RTX 5060 8GB) 실측 응답 시간으로 1대가 한 건씩 처리할 때의 월 처리량(보수적)
                capacity = HOURS_PER_MONTH * 3600 / data["local_latency_sec"]
                for multiplier in CLOUD_PRICE_MULTIPLIERS:
                    per_doc = data["cloud_cost_per_doc"] * multiplier
                    docs = monthly / per_doc
                    results.append({"server": key, "with_labor": with_labor,
                                    "scenario": scenario, "cloud_price_multiplier": multiplier,
                                    "monthly_cost_usd": monthly, "power_cost_usd": power,
                                    "cloud_cost_per_1000_docs_usd": per_doc * 1000,
                                    "break_even_docs_per_month": docs,
                                    "capacity_docs_per_month_laptop": capacity,
                                    "servers_needed_at_break_even": docs / capacity})
    write_json(OUTPUT / "break_even.json", {
        "created_at": now_iso(), "inputs": [str(p.relative_to(ROOT)) for p in inputs],
        "local_target": LOCAL_TARGET, "cloud_target": CLOUD_TARGET,
        "assumptions": {"krw_per_usd": KRW_PER_USD, "electricity_krw_per_kwh":
                        ELECTRICITY_KRW_PER_KWH, "depreciation_months": DEPRECIATION_MONTHS,
                        "maintenance_rate_per_year": MAINTENANCE_RATE_PER_YEAR,
                        "ops_labor_usd_per_month": OPS_LABOR_USD_PER_MONTH,
                        "hours_per_month": HOURS_PER_MONTH, "servers": SERVERS,
                        "power_note": "24시간 최대 소비전력으로 계산(상한)",
                        "capacity_note": "노트북 RTX 5060 실측 응답 시간, 한 번에 한 건 처리(보수적)"},
        "scenarios": rows, "results": results, "sources": SOURCES})

    def fmt(value):
        return f"{value / 10000:,.1f}만" if value >= 10000 else f"{value:,.0f}"

    lines = ["# 손익분기: 로컬 GPU 서버 vs 상용 API", "",
             f"`scripts/break_even.py`가 생성. 생성 시각 {now_iso()}.", "",
             (f"로컬 {LOCAL_TARGET}, 상용 gpt-6-luna({CLOUD_TARGET}). **상용 비용과 응답 시간은 실측**, "
             "서버 장비값·전력·유지보수·인력은 **추정**이다."), "",
             "## 문서 길이별 실측", "",
             "| 문서 | 상용 입력 토큰 | 상용 1,000건 비용 | 로컬 1건 응답 시간 | 노트북 1대 월 처리량 |",
             "|---|---|---|---|---|"]
    for data in rows.values():
        lines.append(f"| {data['label']} | {data['prompt_tokens']:,.0f} | "
                     f"${data['cloud_cost_per_doc'] * 1000:.3f} | {data['local_latency_sec']:.2f}초 | "
                     f"{fmt(HOURS_PER_MONTH * 3600 / data['local_latency_sec'])}건 |")
    lines += ["", "## 서버 가정 (추정)", "",
              "| 구성 | 장비값 | 월 고정비(인력 제외) | 그중 전기료 | 근거 |", "|---|---|---|---|---|"]
    for key, server in SERVERS.items():
        monthly, power = monthly_cost(server, False)
        lines.append(f"| {server['name']} | ${server['price_usd']:,} | ${monthly:,.0f} | "
                     f"${power:,.0f} | {server['basis']} |")
    lines += ["", (f"공통 가정: 3년 감가상각, 유지보수 연 10%, 전기 {ELECTRICITY_KRW_PER_KWH}원/kWh"
              f"(24시간 최대 소비전력, 상한), 환율 {KRW_PER_USD}원/$. 관리 인력은 회사마다 달라 "
              f"월 ${OPS_LABOR_USD_PER_MONTH:,}를 더한 경우를 따로 보였다."), "",
              "## 월 몇 건부터 로컬이 싼가", "",
              "| 구성 | 문서 | 지금 단가 | 지금 단가 + 인력 | 단가 10배 모델 | 단가 10배 + 인력 |",
              "|---|---|---|---|---|---|"]
    for key, server in SERVERS.items():
        for scenario, data in rows.items():
            cells = []
            for multiplier in CLOUD_PRICE_MULTIPLIERS:
                for with_labor in (False, True):
                    hit = next(r for r in results if r["server"] == key and r["scenario"] == scenario
                               and r["with_labor"] == with_labor
                               and r["cloud_price_multiplier"] == multiplier)
                    cells.append(f"{fmt(hit['break_even_docs_per_month'])}건"
                                 + (" ⚠" if hit["servers_needed_at_break_even"] > 1 else ""))
            lines.append(f"| {server['name']} | {data['label'].split(',')[0]} | " + " | ".join(cells) + " |")
    lines += ["", ("⚠: 손익분기 건수가 노트북 GPU 1대의 월 처리량(한 번에 한 건 기준)을 넘는다. "
              "실제 서버 GPU는 더 빠르고 동시 처리도 되므로 처리량은 이보다 크지만, 측정하지 않았다."), "",
              "## 출처 (2026-10-13 확인)", ""] + [f"- {url}" for url in SOURCES]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
