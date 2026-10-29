"""UltraEdit 的逐题答案 token 正确比例及独立的补充指标。"""


def token_accuracy(predicted, target):
    if not target or len(predicted) != len(target):
        raise ValueError("短答案 token accuracy 要求非空且等长的预测和标签")
    return sum(a == b for a, b in zip(predicted, target)) / len(target)


def paired_scores(before, after, target):
    return {
        "token_accuracy": {"before": token_accuracy(before, target), "after": token_accuracy(after, target)},
        "exact_match": {"before": float(tuple(before) == tuple(target)),
                        "after": float(tuple(after) == tuple(target))},
        "output_token_agreement": token_accuracy(after, before),
        "output_exact_agreement": float(tuple(before) == tuple(after)),
    }
