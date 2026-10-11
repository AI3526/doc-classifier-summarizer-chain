"""문자 TF-IDF와 로지스틱 회귀로 비교 기준 모델을 학습한다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step03_train_baseline.py
uv run python steps/step03_train_baseline.py --ngram 1 4 --run-name baseline_char1-4
"""

import argparse
import time
from pathlib import Path

import joblib
from common.metrics import classification_scores, save_errors
from common.records import new_run_dir, now_iso, write_json
from common.resources import environment_info, measure_latency, process_rss_mb
from lesson_settings import DATASET, LEARNER_DIR, ROOT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from step01_read_data import read_labels, read_rows
from step02_check_data import check_splits
from step06_predict import load_run, predict_text

DATA_DIR = ROOT / "data" / DATASET
NGRAM_RANGE = (2, 5)
SEED = 42


def train_baseline(data_dir=DATA_DIR, learner_dir=LEARNER_DIR, run_name=None,
                   ngram_range=NGRAM_RANGE, seed=SEED):
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    ngram_range = tuple(ngram_range)
    run_name = run_name or f"baseline_char{ngram_range[0]}-{ngram_range[1]}"
    labels = read_labels(data_dir)
    train = read_rows(data_dir / "train.csv")
    validation = read_rows(data_dir / "validation.csv")
    check_splits({"train": train, "validation": validation}, labels)
    label2id = {label: index for index, label in enumerate(labels)}

    # fit은 train에서만 수행한다. 어휘와 IDF도 이 단계에서 train으로 학습된다.
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=ngram_range, min_df=1),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    started = time.perf_counter()
    model.fit([row["text"] for row in train], [label2id[row["label"]] for row in train])
    train_time = time.perf_counter() - started
    predictions = model.predict([row["text"] for row in validation]).tolist()
    actual = [label2id[row["label"]] for row in validation]
    metrics = classification_scores(actual, predictions, labels)

    output = new_run_dir(learner_dir, run_name)
    joblib.dump(model, output / "baseline.joblib")
    config = {"kind": "baseline", "dataset": data_dir.name, "learner": learner_dir.name,
              "run_name": run_name, "model": "tfidf-char+logistic-regression",
              "labels": labels, "ngram_range": list(ngram_range), "seed": seed}
    write_json(output / "config.json", config)
    write_json(output / "validation_metrics.json", metrics)
    save_errors(output / "validation_errors.csv", validation, predictions, labels)

    # 서비스와 같은 경로(저장 파일을 다시 불러와 1건씩 예측)로 추론 시간을 잰다.
    bundle = load_run(output)
    latency = measure_latency(lambda text: predict_text(text, bundle),
                              [row["text"] for row in validation])
    vectorizer = model.named_steps["tfidfvectorizer"]
    experiment = {
        "experiment_id": output.name, "learner": learner_dir.name, "dataset": data_dir.name,
        "kind": "baseline",
        "model_info": {
            "model_id": "TF-IDF(char) + LogisticRegression",
            "model_size_params": int(model.named_steps["logisticregression"].coef_.size),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "model_file_mb": (output / "baseline.joblib").stat().st_size / 2**20,
            "precision": "FP64", "training_type": "train 데이터로 처음부터 학습",
            "pretrained_source": None,
        },
        "hyperparameters": {"ngram_range": list(ngram_range), "analyzer": "char",
                            "class_weight": "balanced", "max_iter": 1000, "seed": seed},
        "resource_and_time": {
            "train_time_sec": train_time,
            "inference_latency_per_sample_ms": latency["mean_ms"],
            "inference_latency": {**latency, "device": "cpu"},
            "peak_vram_mb": None, "process_rss_mb": process_rss_mb(),
        },
        "evaluation_metrics": {"split": "validation", "accuracy": metrics["accuracy"],
                               "macro_f1": metrics["macro_f1"],
                               "weighted_f1": metrics["weighted_f1"],
                               "confusion_matrix": metrics["confusion_matrix"]},
        "environment": environment_info(),
        "timestamp": now_iso(),
    }
    write_json(output / "experiment.json", experiment)
    print("기준 모델 validation macro F1:", round(metrics["macro_f1"], 4))
    print(f"학습 {train_time:.2f}초, 1건 추론 평균 {latency['mean_ms']:.2f}ms (CPU)")
    print("저장 위치:", output)
    return output


def main():
    parser = argparse.ArgumentParser(description="TF-IDF + 로지스틱 회귀 기준 모델 학습")
    parser.add_argument("--ngram", type=int, nargs=2, default=NGRAM_RANGE,
                        metavar=("MIN", "MAX"), help="문자 n-gram 범위")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-name", help="실행 이름. 생략하면 n-gram 범위로 정한다.")
    args = parser.parse_args()
    train_baseline(run_name=args.run_name, ngram_range=args.ngram, seed=args.seed)


if __name__ == "__main__":
    main()
