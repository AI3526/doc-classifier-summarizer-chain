"""요약 품질 지표 계산. 사람 채점과 LLM 채점에 같은 공식을 쓴다.

주 지표는 사용 가능률(그대로 보여 줘도 되는 요약의 비율)이다.
100점 감점 점수는 같은 체크리스트에서 계산하는 발표용 요약 숫자다.
"""

import json
import re
import statistics

# 사용 가능 기준: 형식 정상, 환각·수치 오류·외국어 없음, 핵심 포함률이 이 값 이상
USABLE_MIN_COVERAGE = 0.67
POINT_VALUE = {"included": 1.0, "partial": 0.5, "missing": 0.0}
# 모든 항목이 최대로 감점되면 정확히 100이 되도록 맞췄다(30+20+15+5+5+25).
DEDUCTIONS = {"hallucination": 30, "numeric_error": 20, "foreign_language": 15,
              "policy_violation": 5, "awkward_sentence": 5,
              "missing_point": 8, "partial_point": 4, "coverage_cap": 25}
# LLM 채점을 믿고 쓸지 정하는 기준: 사람 다수결과의 사용 가능 판정 일치도.
# 판정이 한쪽으로 SKEW_RATIO보다 많이 몰리면 kappa가 낮게 나오는 문제가 있어 PABAK으로 판단한다.
AGREEMENT_THRESHOLD = 0.7
SKEW_RATIO = 0.85
BINARY_FIELDS = ("hallucination", "numeric_error", "foreign_language", "policy_violation",
                 "awkward_sentence")

_HAN_OR_KANA = re.compile(r"[぀-ヿ一-鿿]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def detect_foreign(summary, source_text):
    """요약 값에 원문에 없는 한자·가나·영문 단어가 있으면 True. JSON 키는 보지 않는다."""
    values = json.dumps(list(summary.values()) if isinstance(summary, dict) else summary,
                        ensure_ascii=False)
    if any(char not in source_text for char in _HAN_OR_KANA.findall(values)):
        return True
    return any(word.lower() not in source_text.lower() for word in _LATIN_WORD.findall(values))


def score_checklist(checklist, format_ok=True, foreign_auto=False):
    """체크리스트 하나를 지표로 바꾼다. 형식이 깨진 응답은 0점, 사용 불가로 둔다."""
    if not format_ok or checklist is None:
        return {"format_ok": False, "usable": False, "coverage": 0.0, "hallucination": None,
                "numeric_error": None, "foreign_language": None, "policy_violation": None,
                "awkward_sentence": None, "deduction_score": 0}
    statuses = [item["status"] for item in checklist["required_points"]]
    coverage = (statistics.mean(POINT_VALUE[status] for status in statuses)
                if statuses else 1.0)
    flags = {
        "hallucination": bool(checklist["unsupported_claims"])
        or any(item["violated"] for item in checklist["forbidden_claims"]),
        "numeric_error": bool(checklist["numeric_error"]),
        "foreign_language": bool(checklist["foreign_language"]) or foreign_auto,
        "policy_violation": bool(checklist["policy_violation"]),
        "awkward_sentence": bool(checklist["awkward_sentence"]),
    }
    point_penalty = min(DEDUCTIONS["coverage_cap"],
                        DEDUCTIONS["missing_point"] * statuses.count("missing")
                        + DEDUCTIONS["partial_point"] * statuses.count("partial"))
    score = max(0, 100 - point_penalty - sum(DEDUCTIONS[key] for key, on in flags.items() if on))
    usable = (not flags["hallucination"] and not flags["numeric_error"]
              and not flags["foreign_language"] and coverage >= USABLE_MIN_COVERAGE)
    return {"format_ok": True, "usable": usable, "coverage": coverage, **flags,
            "deduction_score": score}


def summarize_scores(rows):
    """여러 응답의 지표를 모델 단위로 묶는다."""
    if not rows:
        return {}
    parsed = [row for row in rows if row["format_ok"]]

    def rate(key, items):
        values = [row[key] for row in items if row[key] is not None]
        return sum(values) / len(values) if values else None

    return {"n": len(rows), "usable_rate": rate("usable", rows),
            "coverage_mean": statistics.mean(row["coverage"] for row in rows),
            "hallucination_rate": rate("hallucination", parsed),
            "numeric_error_rate": rate("numeric_error", parsed),
            "foreign_language_rate": rate("foreign_language", parsed),
            "policy_violation_rate": rate("policy_violation", parsed),
            "awkward_sentence_rate": rate("awkward_sentence", parsed),
            "deduction_score_mean": statistics.mean(row["deduction_score"] for row in rows),
            "format_fail_rate": 1 - len(parsed) / len(rows)}


# 생성 모델 최종 선택 규칙(평가 기준 8장): 로컬 우선 + 품질 하한
LOCAL_MAX_USABLE_GAP = 0.10  # 로컬 사용 가능률이 상용보다 이만큼까지 낮아도 허용
LOCAL_MAX_HALLUCINATION_GAP = 0.10  # 로컬 환각률이 상용보다 이만큼까지 높아도 허용


def choose_generation_model(targets, model_types, latency):
    """targets: 모델별 summarize_scores 결과, model_types: local/cloud, latency: 평균 응답 시간(s)."""
    clouds = [alias for alias in targets if model_types[alias] == "cloud"]
    locals_ = [alias for alias in targets if model_types[alias] == "local"]
    if len(clouds) != 1:
        raise ValueError("상용 기준 모델이 하나 있어야 합니다.")
    cloud = targets[clouds[0]]
    checks = {}
    for alias in locals_:
        row = targets[alias]
        usable_gap = cloud["usable_rate"] - row["usable_rate"]
        hallucination_gap = (row["hallucination_rate"] or 0) - (cloud["hallucination_rate"] or 0)
        checks[alias] = {"usable_gap": usable_gap, "hallucination_gap": hallucination_gap,
                         "passes": usable_gap <= LOCAL_MAX_USABLE_GAP
                         and hallucination_gap <= LOCAL_MAX_HALLUCINATION_GAP}
    passing = [alias for alias in locals_ if checks[alias]["passes"]]
    if passing:
        chosen = min(passing, key=lambda a: (-targets[a]["usable_rate"],
                                             targets[a]["hallucination_rate"] or 0, latency[a]))
        reason = "품질 하한을 만족하는 로컬 모델 중 사용 가능률 → 환각률 → 응답 시간 순"
    else:
        chosen = clouds[0]
        reason = "품질 하한을 만족하는 로컬 모델이 없어 상용 모델 선택"
    return {"selected": chosen, "model_type": model_types[chosen], "reason": reason,
            "cloud_reference": clouds[0], "local_checks": checks,
            "rule": {"local_max_usable_gap": LOCAL_MAX_USABLE_GAP,
                     "local_max_hallucination_gap": LOCAL_MAX_HALLUCINATION_GAP}}


def cohen_kappa(a, b):
    """두 채점자의 예/아니오(또는 범주) 일치도. 우연히 일치할 확률을 뺀 값."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return None
    observed = sum(x == y for x, y in pairs) / len(pairs)
    labels = {x for pair in pairs for x in pair}
    expected = sum((sum(x == label for x, _ in pairs) / len(pairs))
                   * (sum(y == label for _, y in pairs) / len(pairs)) for label in labels)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def pabak(a, b):
    """한쪽 판정이 많을 때 덜 흔들리는 일치도. 2 × 관측 일치율 − 1 (예/아니오 판정)."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return None
    return 2 * sum(x == y for x, y in pairs) / len(pairs) - 1


def llm_trust_decision(human, llm):
    """사용 가능 판정의 사람-LLM 일치도로 LLM 채점을 믿고 쓸지 정한다(평가 기준 7장)."""
    pooled = [value for value in (*human, *llm) if value is not None]
    positive = sum(pooled) / len(pooled) if pooled else None
    skewed = positive is not None and max(positive, 1 - positive) > SKEW_RATIO
    kappa, adjusted = cohen_kappa(human, llm), pabak(human, llm)
    criterion, value = ("pabak", adjusted) if skewed else ("cohen_kappa", kappa)
    return {"usable_positive_ratio": positive, "skewed": skewed, "criterion": criterion,
            "value": value, "threshold": AGREEMENT_THRESHOLD,
            "use_llm_scores": value is not None and value >= AGREEMENT_THRESHOLD,
            "cohen_kappa": kappa, "pabak": adjusted}


def fleiss_kappa(ratings):
    """여러 채점자의 일치도. ratings: 항목마다 채점자들의 판정 목록."""
    items = [[value for value in row if value is not None] for row in ratings]
    items = [row for row in items if len(row) >= 2]
    if not items:
        return None
    labels = sorted({value for row in items for value in row}, key=str)
    per_item, totals = [], {label: 0 for label in labels}
    for row in items:
        n = len(row)
        counts = {label: row.count(label) for label in labels}
        for label in labels:
            totals[label] += counts[label]
        per_item.append((sum(c * c for c in counts.values()) - n) / (n * (n - 1)))
    all_ratings = sum(totals.values())
    expected = sum((count / all_ratings) ** 2 for count in totals.values())
    observed = statistics.mean(per_item)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def pearson(a, b):
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs, strict=True)
    if statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return None
    return statistics.correlation(xs, ys)
