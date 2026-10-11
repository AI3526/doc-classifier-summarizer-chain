"""실험 기록(JSON)을 모아 보고서용 표와 그림을 만든다. 모델을 학습하거나 호출하지 않는다.

uv run python scripts/build_report.py
결과: reports/figures/classification/*.png, reports/generated/classification_tables.md
그림의 축 이름은 한글 글꼴 문제를 피하려고 영어로 쓴다.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

import matplotlib.pyplot as plt
from common.plots import plot_confusion_matrix
from common.records import now_iso, read_json
from lesson_settings import DATASET, OUTPUT_ROOT, SELECTION_DIR
from step05_select_model import LATENCY_TIE_RATIO, MIN_SEEDS, TOLERANCE, cell

FIGURES = ROOT / "reports" / "figures" / "classification"
GENERATED = ROOT / "reports" / "generated"
MODEL_ORDER = ["baseline", "electra-s", "bert", "roberta", "electra-b", "deberta-ko"]
COLORS = {"baseline": "#7f7f7f", "electra-s": "#2ca02c", "bert": "#1f77b4",
          "roberta": "#d62728", "electra-b": "#9467bd", "deberta-ko": "#ff7f0e"}


def order(alias):
    return MODEL_ORDER.index(alias) if alias in MODEL_ORDER else len(MODEL_ORDER)


def lr_label(value):
    return "-" if value is None else f"{value:.0e}".replace("e-0", "e-")


def best_group_per_model(groups):
    """모델마다 평균 macro F1이 가장 높은 설정 하나(같으면 seed 수가 많은 쪽)."""
    best = {}
    for group in groups:
        key = (group["macro_f1_mean"], group["n_seeds"])
        current = best.get(group["model_alias"])
        if current is None or key > (current["macro_f1_mean"], current["n_seeds"]):
            best[group["model_alias"]] = group
    return sorted(best.values(), key=lambda group: order(group["model_alias"]))


def draw_groups(axis, groups, selected_group, size_scale):
    for group in groups:
        alias = group["model_alias"]
        size = (40 + group["params"] / 1e6 * 2.5) * size_scale
        marker = "s" if group["kind"] == "baseline" else "o"
        axis.errorbar(group["cpu_latency_ms_mean"], group["macro_f1_mean"],
                      yerr=group["macro_f1_std"] or 0, fmt="none", ecolor=COLORS.get(alias),
                      alpha=0.6, capsize=3)
        axis.scatter(group["cpu_latency_ms_mean"], group["macro_f1_mean"], s=size,
                     marker=marker, color=COLORS.get(alias), alpha=0.75,
                     edgecolors="black" if group["n_seeds"] >= MIN_SEEDS else "none")
    axis.scatter(selected_group["cpu_latency_ms_mean"], selected_group["macro_f1_mean"],
                 s=380 * size_scale, marker="*", color="black", zorder=5)


def plot_speed_vs_f1(groups, selected_group, path):
    best_f1 = max(group["macro_f1_mean"] for group in groups)
    band = [group for group in groups if best_f1 - group["macro_f1_mean"] <= TOLERANCE]
    fastest = min(group["cpu_latency_ms_mean"] for group in band)
    fig, (overview, zoom) = plt.subplots(1, 2, figsize=(14, 5.5),
                                         gridspec_kw={"width_ratios": [1, 1.1]})

    # 왼쪽: 전체 설정. 기준 모델부터 사전학습 모델까지 속도 차이를 로그 축으로 본다.
    overview.axhspan(best_f1 - TOLERANCE, best_f1 + 0.002, color="gold", alpha=0.25)
    draw_groups(overview, groups, selected_group, 1.0)
    overview.set_xscale("log")
    low = min(g["macro_f1_mean"] for g in groups if g["macro_f1_mean"] > 0.9)
    overview.set_ylim(min(0.93, low - 0.01), 1.01)
    overview.set(xlabel="CPU latency per sample (ms, log scale)",
                 ylabel="validation macro F1 (seed mean)",
                 title="All configurations (F1 > 0.9 shown)")
    for alias in MODEL_ORDER:
        if any(group["model_alias"] == alias for group in groups):
            overview.scatter([], [], color=COLORS[alias], label=alias,
                             marker="s" if alias == "baseline" else "o")
    overview.scatter([], [], marker="*", color="black", s=120, label="selected")
    overview.fill_between([], [], color="gold", alpha=0.25, label=f"tie band (best - {TOLERANCE})")
    overview.legend(fontsize=8, loc="lower right")
    overview.grid(alpha=0.3, which="both")

    # 오른쪽: 동점 범위를 확대해 선정 규칙(같은 F1, 같은 속도 구간)을 보여 준다.
    zoom.axhspan(best_f1 - TOLERANCE, best_f1 + 0.002, color="gold", alpha=0.25)
    zoom.axvspan(fastest, fastest * LATENCY_TIE_RATIO, color="tab:blue", alpha=0.08,
                 label=f"same-speed zone (fastest x {LATENCY_TIE_RATIO})")
    near = [group for group in groups
            if group["macro_f1_mean"] >= best_f1 - 2 * TOLERANCE and group["kind"] != "baseline"]
    draw_groups(zoom, near, selected_group, 1.4)
    # 같은 높이(F1)에 모인 설정은 이름표를 오른쪽에 위아래로 나눠 놓고 선으로 잇는다.
    clusters = {}
    for group in near:
        clusters.setdefault(round(group["macro_f1_mean"], 4), []).append(group)
    for y_value, members in clusters.items():
        members.sort(key=lambda group: group["cpu_latency_ms_mean"])
        right = max(group["cpu_latency_ms_mean"] for group in members) + 1.5
        for k, group in enumerate(members):
            label_y = y_value + ((len(members) - 1) / 2 - k) * 0.0014
            zoom.annotate(f"{group['model_alias']} {lr_label(group['learning_rate'])} "
                          f"(n={group['n_seeds']})",
                          (group["cpu_latency_ms_mean"], group["macro_f1_mean"]),
                          xytext=(right, label_y), textcoords="data", fontsize=7.5,
                          va="center",
                          arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.6})
    zoom.set(xlabel="CPU latency per sample (ms)", ylabel="validation macro F1 (seed mean)",
             title="Zoom on the tie band (black edge = 3 seeds, bar = seed std)")
    zoom.grid(alpha=0.3)
    zoom.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_lr_heatmap(groups, path):
    models = sorted({g["model_alias"] for g in groups if g["kind"] == "transformer"}, key=order)
    rates = sorted({g["learning_rate"] for g in groups if g["kind"] == "transformer"})
    lookup = {(g["model_alias"], g["learning_rate"]): g for g in groups}
    matrix = [[lookup[(m, r)]["macro_f1_mean"] if (m, r) in lookup else float("nan")
               for r in rates] for m in models]
    fig, axis = plt.subplots(figsize=(6, 4.2))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    for i, model in enumerate(models):
        for j, rate in enumerate(rates):
            group = lookup.get((model, rate))
            if group:
                axis.text(j, i, f"{group['macro_f1_mean']:.3f}\n(n={group['n_seeds']})",
                          ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(rates)), [lr_label(rate) for rate in rates])
    axis.set_yticks(range(len(models)), models)
    axis.set(xlabel="learning rate", title="Validation macro F1 by model x learning rate")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_resources(best_groups, path):
    names = [group["model_alias"] for group in best_groups]
    colors = [COLORS.get(name) for name in names]
    metrics = [("peak_vram_mb_max", "peak VRAM during training (MB)"),
               ("train_time_sec_mean", "training time, 5 epochs (s)"),
               ("cpu_latency_ms_mean", "CPU latency per sample (ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, (key, title) in zip(axes, metrics, strict=True):
        values = [group[key] or 0 for group in best_groups]
        axis.barh(names, values, color=colors)
        for index, value in enumerate(values):
            axis.text(value, index, f" {value:.1f}" if value < 100 else f" {value:.0f}",
                      va="center", fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.3)
        axis.set_xlim(0, max(values) * 1.25 or 1)
    axes[0].axvline(8151, color="red", linestyle="--", linewidth=1)
    axes[0].text(8151, len(names) - 0.5, " GPU 8GB", color="red", fontsize=8)
    axes[0].set_xlim(0, 9000)
    fig.suptitle("Resources of the best configuration per model (baseline: no GPU)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_learning_curves(best_groups, path):
    fig, (f1_axis, loss_axis) = plt.subplots(1, 2, figsize=(11, 4))
    for group in best_groups:
        if group["kind"] != "transformer":
            continue
        run = next(run for run in group["runs"] if run.endswith("_s42")) \
            if any(run.endswith("_s42") for run in group["runs"]) else group["runs"][0]
        epochs = [json.loads(line) for line in
                  (OUTPUT_ROOT / run / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
        label = f"{group['model_alias']} lr={lr_label(group['learning_rate'])}"
        color = COLORS.get(group["model_alias"])
        x = [row["epoch"] for row in epochs]
        f1_axis.plot(x, [row["validation_macro_f1"] for row in epochs], marker="o",
                     color=color, label=label)
        loss_axis.plot(x, [row["validation_loss"] for row in epochs], marker="o", color=color)
        loss_axis.plot(x, [row["train_loss"] for row in epochs], linestyle=":", color=color)
    f1_axis.set(xlabel="epoch", ylabel="validation macro F1", title="Validation macro F1 (seed 42)")
    loss_axis.set(xlabel="epoch", ylabel="loss",
                  title="Loss (solid = validation, dotted = train)")
    for axis in (f1_axis, loss_axis):
        axis.grid(alpha=0.3)
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    f1_axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_seed_spread(groups, candidates, path):
    repeated = sorted((g for g in groups if g["n_seeds"] >= MIN_SEEDS),
                      key=lambda g: -g["macro_f1_mean"])
    by_run = {row["run_dir"]: row for row in candidates}
    fig, axis = plt.subplots(figsize=(8, 3.8))
    for index, group in enumerate(repeated):
        scores = [by_run[run]["validation_macro_f1"] for run in group["runs"]]
        # 같은 점수의 seed가 한 점으로 겹치지 않게 좌우로 조금씩 벌린다.
        offsets = [(i - (len(scores) - 1) / 2) * 0.12 for i in range(len(scores))]
        axis.scatter([index + offset for offset in offsets], scores,
                     color=COLORS.get(group["model_alias"]), s=50, alpha=0.8)
        axis.hlines(group["macro_f1_mean"], index - 0.25, index + 0.25, color="black")
    axis.set_xticks(range(len(repeated)),
                    [f"{g['model_alias']}\n{lr_label(g['learning_rate'])}" for g in repeated],
                    fontsize=8)
    axis.set(ylabel="validation macro F1", title="Seed variation (seeds 42/43/44, line = mean)")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown(rows, columns, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(row.get(key)) for key in columns) + " |" for row in rows]
    return lines


def write_tables(selection, best_groups, path):
    groups = sorted(selection["groups"], key=lambda g: (-g["macro_f1_mean"], order(g["model_alias"])))
    for group in groups:
        group["lr"] = lr_label(group["learning_rate"])
        group["params_m"] = group["params"] / 1e6
        group["mark"] = "✔" if group["config_key"] == selection["selected_group"]["config_key"] else ""
    for group in best_groups:
        group["lr"] = lr_label(group["learning_rate"])
        group["params_m"] = group["params"] / 1e6
    runs = sorted(selection["candidates"], key=lambda r: (order(r["model_alias"]),
                                                          r["learning_rate"] or 0, r["seed"]))
    for run in runs:
        run["lr"] = lr_label(run["learning_rate"])
    lines = [f"# 분류 실험 표 ({DATASET})", "",
             f"`scripts/build_report.py`가 생성. 생성 시각 {now_iso()}. 직접 수정하지 마세요.", "",
             "## 모델별 최고 설정", "",
             *markdown(best_groups,
                       ["model_alias", "model", "params_m", "lr", "n_seeds", "macro_f1_mean",
                        "macro_f1_std", "train_time_sec_mean", "peak_vram_mb_max",
                        "cpu_latency_ms_mean"],
                       ["모델", "ID", "파라미터(M)", "lr", "seed 수", "val Macro-F1 평균",
                        "표준편차", "학습 시간(s)", "peak VRAM(MB)", "CPU 추론(ms)"]),
             "", "## 설정별 (seed 평균)", "",
             *markdown(groups,
                       ["mark", "model_alias", "lr", "n_seeds", "macro_f1_mean", "macro_f1_std",
                        "accuracy_mean", "train_time_sec_mean", "peak_vram_mb_max",
                        "cpu_latency_ms_mean"],
                       ["선정", "모델", "lr", "seed 수", "Macro-F1 평균", "표준편차",
                        "Accuracy 평균", "학습 시간(s)", "peak VRAM(MB)", "CPU 추론(ms)"]),
             "", "## 실행별 전체", "",
             *markdown(runs,
                       ["run_name", "model_alias", "lr", "seed", "best_epoch",
                        "validation_accuracy", "validation_macro_f1", "train_time_sec",
                        "cpu_latency_ms", "peak_vram_mb"],
                       ["실행", "모델", "lr", "seed", "최고 epoch", "Accuracy", "Macro-F1",
                        "학습 시간(s)", "CPU 추론(ms)", "peak VRAM(MB)"])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    selection = read_json(SELECTION_DIR / "selected.json")
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    groups = selection["groups"]
    best_groups = best_group_per_model(groups)

    plot_speed_vs_f1(groups, selection["selected_group"], FIGURES / "speed_vs_f1.png")
    plot_lr_heatmap(groups, FIGURES / "lr_heatmap.png")
    plot_resources(best_groups, FIGURES / "resources.png")
    plot_learning_curves(best_groups, FIGURES / "learning_curves.png")
    plot_seed_spread(groups, selection["candidates"], FIGURES / "seed_spread.png")
    for name in ("selected", "baseline_reference"):
        row = selection[name]
        metrics = read_json(OUTPUT_ROOT / row["run_dir"] / "validation_metrics.json")
        plot_confusion_matrix(metrics["confusion_matrix"], row["labels"],
                              FIGURES / f"confusion_validation_{name}.png",
                              f"{row['model_alias']} (validation)")
    test_path = SELECTION_DIR / "test_metrics.json"
    if test_path.exists():
        test = read_json(test_path)
        for name in ("selected", "baseline_reference"):
            plot_confusion_matrix(test[name]["scores"]["confusion_matrix"],
                                  selection[name]["labels"],
                                  FIGURES / f"confusion_test_{name}.png",
                                  f"{selection[name]['model_alias']} (test)")
    write_tables(selection, best_groups, GENERATED / "classification_tables.md")
    print("그림:", FIGURES)
    print("표:", GENERATED / "classification_tables.md")
    print(f"규칙 상수: TOLERANCE={TOLERANCE}, MIN_SEEDS={MIN_SEEDS}, "
          f"LATENCY_TIE_RATIO={LATENCY_TIE_RATIO}")


if __name__ == "__main__":
    main()
