"""생성 모델 비교 결과로 보고서용 그림과 표를 만든다. build_report.py가 호출한다.

1차(development) 비교들과 가장 최근 2차(eval) 비교를 읽는다. 모델을 호출하지 않는다.
사람 채점은 quality_human.json(실제)을 우선 쓰고, 없으면 quality_human_dummy.json(가상)을 쓰며
가상이면 모든 표와 그림에 표시한다.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from common.records import now_iso
from lesson_settings import LEARNER_DIR, SELECTION_DIR

COMPARISONS = LEARNER_DIR / "comparisons"
GEN_COLORS = {"qwen3-4b": "#17becf", "gemma3-4b": "#2ca02c", "qwen2.5-7b": "#1f77b4",
              "exaone3.5-7.8b": "#ff7f0e", "llama3.1-8b": "#9467bd", "openai": "#d62728"}
GEN_ORDER = list(GEN_COLORS)
DUMMY_MARK = "⚠ 가상 채점(미리보기용)"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def comparison_runs():
    runs = []
    for summary_path in sorted(COMPARISONS.glob("*/summary.json")):
        directory = summary_path.parent
        summary = _load(summary_path)
        quality = _load(directory / "quality_llm.json")
        if not quality:
            continue
        version = summary["prompt_version"].replace("document_summary_", "")
        label = version + ("+S" if summary.get("structured_output") else "")
        same = next((run for run in runs if run["split"] == summary["split"] == "development"
                     and run["label"] == label), None)
        if same:
            # 같은 프롬프트로 모델만 나눠 돌린 개발 비교(예: 로컬과 gpt-6-luna)는 한 묶음으로 본다.
            same["quality"]["targets"].update(quality["targets"])
            same["summary"]["targets"].update(summary["targets"])
            same["dirs"].append(directory)
            continue
        runs.append({"dir": directory, "dirs": [directory], "summary": summary, "quality": quality,
                     "label": label, "split": summary["split"]})
    return runs


def human_quality(directory):
    real = _load(directory / "quality_human.json")
    return (real, False) if real else (_load(directory / "quality_human_dummy.json"), True)


def ordered(aliases):
    return sorted(aliases, key=lambda a: GEN_ORDER.index(a) if a in GEN_ORDER else 99)


def plot_round1(runs, path):
    labels = [run["label"] for run in runs]
    aliases = ordered({a for run in runs for a in run["quality"]["targets"]})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    def series(alias, key):
        values = [run["quality"]["targets"].get(alias, {}).get(key) for run in runs]
        return [float("nan") if value is None else value for value in values]

    for alias in aliases:
        style = {"marker": "s" if alias == "openai" else "o", "color": GEN_COLORS.get(alias),
                 "linestyle": "--" if alias == "openai" else "-", "label": alias}
        axes[0].plot(labels, series(alias, "usable_rate"), **style)
        axes[1].plot(labels, series(alias, "usable_reading_rate"), **style)
    axes[0].set(title="Development: usable rate (incl. field errors)", ylabel="usable rate",
                ylim=(0, 1))
    axes[1].set(title="Development: usable rate for reading (field errors ignored)",
                ylabel="usable rate (reading)", ylim=(0, 1))
    for axis in axes:
        axis.set_xlabel("prompt version (+S = structured output)")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_round2_quality(final, human, is_dummy, path):
    targets = final["quality"]["targets"]
    aliases = ordered(targets)
    metrics = [("usable_rate", "usable"), ("usable_reading_rate", "usable (reading)"),
               ("coverage_mean", "coverage"), ("hallucination_rate", "hallucination"),
               ("field_error_rate", "field error")]
    fig, axis = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(aliases)
    for i, alias in enumerate(aliases):
        xs = [m + (i - (len(aliases) - 1) / 2) * width for m in range(len(metrics))]
        values = [targets[alias][key] or 0 for key, _ in metrics]
        axis.bar(xs, values, width, color=GEN_COLORS.get(alias), label=f"{alias} (LLM judge)")
        for x, value in zip(xs, values, strict=True):
            axis.text(x, value + 0.01, f"{value:.2f}", ha="center", fontsize=7)
        if human and alias in human["targets"]:
            axis.scatter(xs[0], human["targets"][alias]["usable_rate"], marker="D", s=40,
                         color="black", zorder=5,
                         label=("human majority" + (" (DUMMY)" if is_dummy else "")) if i == 0
                         else None)
    axis.set_xticks(range(len(metrics)), [name for _, name in metrics])
    axis.set(ylim=(0, 1.1), ylabel="rate",
             title="Final comparison (eval) summary quality" + (" — human = DUMMY preview" if is_dummy
                                                       else ""))
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_round2_tradeoff(final, selected, path):
    ops, targets = final["summary"]["targets"], final["quality"]["targets"]
    fig, axis = plt.subplots(figsize=(7.5, 5))
    for alias in ordered(targets):
        vram = ops[alias]["vram_mb_max"]
        size = 80 + (vram or 0) / 25
        marker = "s" if ops[alias]["model_type"] == "cloud" else "o"
        axis.scatter(ops[alias]["latency_mean_sec"], targets[alias]["usable_rate"], s=size,
                     marker=marker, color=GEN_COLORS.get(alias), alpha=0.8,
                     edgecolors="black" if alias == selected else "none", linewidths=2)
        cost = ops[alias]["api_cost_per_1000_docs_usd"]
        note = f"{alias}\nVRAM {vram / 1024:.1f}GB" if vram else f"{alias}\n${cost:.3f}/1k docs"
        axis.annotate(note, (ops[alias]["latency_mean_sec"], targets[alias]["usable_rate"]),
                      xytext=(10, -4), textcoords="offset points", fontsize=8)
    # x축을 0부터 그려 몇백 ms 차이가 과장되어 보이지 않게 한다.
    axis.set_xlim(0, max(ops[a]["latency_mean_sec"] for a in targets) * 1.5)
    axis.set(xlabel="mean response time (s)", ylabel="usable rate incl. field errors (LLM judge)",
             title="Final comparison: quality vs speed (square = cloud, black edge = selected)",
             ylim=(0, 1.05))
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _f(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def write_tables(round1, final, human, is_dummy, selection, path):
    lines = ["# 생성 모델 비교 표", "",
             f"`scripts/build_report.py`가 생성. 생성 시각 {now_iso()}. 직접 수정하지 마세요.", ""]
    if round1:
        aliases = ordered({a for run in round1 for a in run["quality"]["targets"]})
        lines += [("## 개발 비교 (development 15건): 사용 가능률 / 읽기용 사용 가능률 / 칸 정확도 / "
                   "형식 실패율"), "",
                  "| 모델 | " + " | ".join(run["label"] for run in round1) + " |",
                  "|---|" + "---|" * len(round1)]
        for alias in aliases:
            cells = []
            for run in round1:
                row = run["quality"]["targets"].get(alias)
                cells.append(f"{_f(row['usable_rate'], 2)} / {_f(row.get('usable_reading_rate'), 2)}"
                             f" / {_f(row.get('field_accuracy_mean'), 2)}"
                             f" / {_f(row['format_fail_rate'], 2)}" if row else "-")
            lines.append(f"| {alias} | " + " | ".join(cells) + " |")
        lines.append("")
    if final:
        ops, targets = final["summary"]["targets"], final["quality"]["targets"]
        s = final["summary"]
        lines += [(f"## 최종 비교 (eval {s['cases']}건 × {s['repeats']}회, 프롬프트 "
                  f"`{s['prompt_version']}`, 구조화 출력 {'사용' if s.get('structured_output') else '미사용'})"),
                  "", "### 품질 (LLM 채점, 전체 응답)", "",
                  ("| 모델 | 유형 | 사용 가능률 | 읽기용 사용 가능률 | 핵심 포함률 | 환각률 | 수치 오류 | "
                   "칸 오류 | 칸 정확도 | 외국어 | 어색한 문장 | 감점 점수 | 형식 실패 |"),
                  "|---|" + "---|" * 12]
        for alias in ordered(targets):
            t = targets[alias]
            lines.append(f"| {alias} | {ops[alias]['model_type']} | {_f(t['usable_rate'])} | "
                         f"{_f(t.get('usable_reading_rate'))} | "
                         f"{_f(t['coverage_mean'])} | {_f(t['hallucination_rate'])} | "
                         f"{_f(t['numeric_error_rate'])} | {_f(t.get('field_error_rate'))} | "
                         f"{_f(t.get('field_accuracy_mean'))} | {_f(t['foreign_language_rate'])} | "
                         f"{_f(t['awkward_sentence_rate'])} | "
                         f"{_f(t['deduction_score_mean'], 1)} | {_f(t['format_fail_rate'])} |")
        lines += ["", ("- 사용 가능률: 칸 오류까지 없는 응답의 비율(자동화 기준, 주 지표). "
                  "읽기용 사용 가능률: 칸 오류를 빼고 판정한 비율(사람이 읽기만 하는 기준)."), ""]
        lines += ["", "### 운영 지표", "",
                  ("| 모델 | 응답 평균(s) | 응답 p95(s) | TTFT(s) | 토큰/s | 입력 토큰 | 출력 토큰 | "
                  "VRAM(MB) | 콜드 스타트(s) | 1,000건 API 비용($) | 오류율 |"), "|---|" + "---|" * 10]
        for alias in ordered(targets):
            o = ops[alias]
            lines.append(f"| {alias} | {_f(o['latency_mean_sec'], 2)} | {_f(o['latency_p95_sec'], 2)}"
                         f" | {_f(o['ttft_mean_sec'], 3)} | {_f(o['tokens_per_second_mean'], 0)} | "
                         f"{_f(o['prompt_tokens_mean'], 0)} | {_f(o['completion_tokens_mean'], 0)} | "
                         f"{_f(o['vram_mb_max'], 0)} | {_f(o['warmup_load_time_sec'], 2)} | "
                         f"{_f(o['api_cost_per_1000_docs_usd'], 4)} | {_f(o['error_rate'], 3)} |")
        lines += ["", ("- 로컬 모델의 API 비용은 0이며 장비·전력·관리 비용은 별도다. "
                  "상용 모델 비용은 실측 토큰 × Standard 단가(2026-10-11 확인)로 계산했다."), ""]
    if human:
        agreement = human["agreement"]
        trust = agreement["llm_trust_decision"]
        mark = f" {DUMMY_MARK}" if is_dummy else ""
        lines += [(f"### 사람 채점 (블라인드 {agreement['items']}건, 채점자 "
                  f"{len(agreement['raters'])}명 다수결){mark}"), "",
                  "| 모델 | n | 사용 가능률 | 핵심 포함률 | 환각률 | 감점 점수 |", "|---|---|---|---|---|---|"]
        for alias in ordered(human["targets"]):
            t = human["targets"][alias]
            lines.append(f"| {alias} | {t['n']} | {_f(t['usable_rate'])} | {_f(t['coverage_mean'])} "
                         f"| {_f(t['hallucination_rate'])} | {_f(t['deduction_score_mean'], 1)} |")
        lines += ["", f"### 일치도{mark}", "",
                  ("| 항목 | 사람-LLM 일치율 | 사람-LLM Cohen's kappa | 사람-LLM PABAK | "
                   "사람 간 Fleiss' kappa | 사람 간 PABAK |"),
                  "|---|---|---|---|---|---|"]
        for field, row in agreement["llm_vs_human"].items():
            if not isinstance(row, dict):
                continue
            lines.append(f"| {field} | {_f(row['agreement'])} | {_f(row['cohen_kappa'])} | "
                         f"{_f(row['pabak'])} | {_f(agreement['fleiss_kappa'].get(field))} | "
                         f"{_f(agreement.get('rater_pabak', {}).get(field))} |")
        lines += ["", (f"- 감점 점수 상관(피어슨): "
                  f"{_f(agreement['llm_vs_human']['deduction_score_pearson'])}"),
                  (f"- LLM 채점 신뢰 판정: 기준 {trust['criterion']} = {_f(trust['value'])} "
                  f"(기준값 {trust['threshold']}, 사용 가능 비율 {_f(trust['usable_positive_ratio'])}"
                  f"{', 치우침 → PABAK' if trust['skewed'] else ''}) → "
                  f"{'LLM 채점 사용' if trust['use_llm_scores'] else '사람 다수결 우선'}"), ""]
    if selection:
        status = {"final": "확정", "provisional": "잠정(LLM 채점만 반영)",
                  "dummy": f"미리보기 {DUMMY_MARK}"}[selection["status"]]
        lines += [f"## 생성 모델 선택: **{selection['selected']}** ({status})", "",
                  f"- 품질 출처: {'LLM 채점' if selection['quality_source'] == 'llm' else '사람 다수결'}",
                  f"- 이유: {selection['reason']}",
                  (f"- 기준: 상용({selection['cloud_reference']}) 대비 사용 가능률 차이 ≤ "
                  f"{selection['rule']['local_max_usable_gap']}, 환각률 차이 ≤ "
                  f"{selection['rule']['local_max_hallucination_gap']}"), "",
                  "| 로컬 모델 | 사용 가능률 차이 (상용 − 로컬) | 환각률 차이 (로컬 − 상용) | 품질 하한 |",
                  "|---|---|---|---|"]
        for alias, check in selection["local_checks"].items():
            lines.append(f"| {alias} | {check['usable_gap']:+.3f} | {check['hallucination_gap']:+.3f}"
                         f" | {'통과' if check['passes'] else '미달'} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(figures_dir, generated_dir):
    runs = comparison_runs()
    round1 = [run for run in runs if run["split"] == "development"]
    finals = [run for run in runs if run["split"] == "eval"]
    final = finals[-1] if finals else None
    figures_dir.mkdir(parents=True, exist_ok=True)
    if round1:
        plot_round1(round1, figures_dir / "round1_versions.png")
    human, is_dummy, selection = None, False, None
    if final:
        human, is_dummy = human_quality(final["dir"])
        selection = (_load(final["dir"] / "generation_selected.json")
                     or _load(final["dir"] / "generation_selected_dummy.json"))
        plot_round2_quality(final, human, is_dummy, figures_dir / "round2_quality.png")
        plot_round2_tradeoff(final, selection["selected"] if selection else None,
                             figures_dir / "round2_tradeoff.png")
    write_tables(round1, final, human, is_dummy, selection, generated_dir / "generation_tables.md")
    return {"round1": len(round1), "final": final["dir"].name if final else None,
            "human_is_dummy": is_dummy if human else None,
            "selection": selection and {k: selection[k] for k in ("selected", "status")}}


if __name__ == "__main__":
    print(SELECTION_DIR)
