"""학습 곡선과 혼동행렬 그림. 한글 글꼴 문제를 피하려고 축 이름은 영어로 쓴다."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장한다.
import matplotlib.pyplot as plt


def plot_learning_curve(epochs_path, output_path, title):
    records = [json.loads(line) for line in
               Path(epochs_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    epochs = [row["epoch"] for row in records]
    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(10, 4))
    loss_ax.plot(epochs, [row["train_loss"] for row in records], marker="o", label="train")
    loss_ax.plot(epochs, [row["validation_loss"] for row in records], marker="o",
                 label="validation")
    loss_ax.set(xlabel="epoch", ylabel="loss", title="Loss")
    loss_ax.legend()
    f1_ax.plot(epochs, [row["validation_macro_f1"] for row in records], marker="o",
               color="tab:green")
    f1_ax.set(xlabel="epoch", ylabel="macro F1", title="Validation macro F1", ylim=(0, 1.02))
    for axis in (loss_ax, f1_ax):
        axis.set_xticks(epochs)
        axis.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(matrix, labels, output_path, title):
    fig, axis = plt.subplots(figsize=(4.5, 4))
    axis.imshow(matrix, cmap="Blues")
    peak = max(max(row) for row in matrix) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axis.text(j, i, str(value), ha="center", va="center",
                      color="white" if value > peak / 2 else "black")
    axis.set_xticks(range(len(labels)), labels, rotation=30)
    axis.set_yticks(range(len(labels)), labels)
    axis.set(xlabel="predicted", ylabel="actual", title=title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
