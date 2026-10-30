"""流式记录与按案例、按问题两种平均口径，保留接受/拒绝分组。"""

from collections import defaultdict
import json


def token_accuracy(predicted, target):
    """s = (1/L) Σ_i 1[ŷ_i = y_i]；要求非空、等长、同位置对齐。"""
    if not target or len(predicted) != len(target):
        raise ValueError("短答案 token accuracy 要求非空且等长的预测和标签")
    return sum(a == b for a, b in zip(predicted, target)) / len(target)


def paired_scores(before, after, target):
    """计算两侧正确率、e = 1[ŷ=y]，以及两侧预测的 token/整段一致率。"""
    before, after, target = tuple(before), tuple(after), tuple(target)
    return {
        "token_accuracy": {"before": token_accuracy(before, target), "after": token_accuracy(after, target)},
        "exact_match": {"before": float(before == target), "after": float(after == target)},
        "output_token_agreement": token_accuracy(after, before),
        "output_exact_agreement": float(before == after),
    }


def write_record(stream, record):
    """写入并刷新一条 JSONL；拒绝把 NaN/Inf 静默保存为数值结果。"""
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def _mean(rows):
    """逐列等权求平均；空集合返回 null，由外层标明覆盖状态。"""
    # 六列依次为前/后正确率、前/后完全匹配率、两种输出一致率。
    return [sum(column) / len(rows) for column in zip(*rows)] if rows else [None] * 6


def _pair(values, offset=0):
    """取一对 before/after，计算 Δ = after - before。"""
    before, after = values[offset:offset + 2]
    return {"before": before, "after": after,
            "delta": after - before if before is not None else None}


class ScoreSummary:
    """按题组、接受状态及源案例累计评分，保留预期和实际题数。"""

    def __init__(self, tasks, metrics):
        """先固定预期分母；每题同时属于 all 和自己的接受/拒绝子集。"""
        self.metrics = metrics
        self.expected = defaultdict(int)
        self.values = defaultdict(lambda: defaultdict(list))
        self.completed = 0
        for task in tasks:
            for subset in ("all", task.decision):
                self.expected[subset, task.group] += 1

    def add(self, task, scores):
        """累计一条已成功保存的配对评分，并增加完成题数。"""
        values = (scores["token_accuracy"]["before"], scores["token_accuracy"]["after"],
                  scores["exact_match"]["before"], scores["exact_match"]["after"],
                  scores["output_token_agreement"], scores["output_exact_agreement"])
        for subset in ("all", task.decision):
            self.values[subset, task.group][task.source_case_id].append(values)
        self.completed += 1

    def export(self):
        """逐题平均 Σ_q s_q/N；逐案例平均 Σ_c(Σ_q∈Q_c s_q/n_c)/C。"""
        result = {}
        for subset in ("all", "accepted", "rejected"):
            result[subset] = {}
            for group, name in self.metrics.items():
                cases = self.values[subset, group]
                rows = [row for probes in cases.values() for row in probes]
                expected = self.expected[subset, group]
                # 保留两种平均顺序；题数不等时，逐案例平均不等于逐题平均。
                per_case = _mean([_mean(probes) for probes in cases.values()])
                per_probe = _mean(rows)
                result[subset][name] = {
                    "group": group, "status": "not_applicable" if expected == 0 else (
                        "complete" if len(rows) == expected else "incomplete"),
                    "expected_tasks": expected, "completed_tasks": len(rows), "case_count": len(cases),
                    "case_macro": _pair(per_case), "probe_macro": _pair(per_probe),
                    "exact_match_case_macro": _pair(per_case, 2),
                    "output_token_agreement_case_macro": per_case[4],
                    "output_exact_agreement_case_macro": per_case[5],
                }
        return result
