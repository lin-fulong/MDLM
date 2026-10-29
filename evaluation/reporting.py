"""流式记录与按案例、按问题两种平均口径，保留接受/拒绝分组。"""

from collections import defaultdict
import json


def write_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def _mean(rows):
    # 六列依次为前/后正确率、前/后完全匹配率、两种输出一致率。
    return [sum(column) / len(rows) for column in zip(*rows)] if rows else [None] * 6


def _pair(values, offset=0):
    before, after = values[offset:offset + 2]
    return {"before": before, "after": after,
            "delta": after - before if before is not None else None}


class ScoreSummary:
    def __init__(self, tasks, metrics):
        self.metrics = metrics
        self.expected = defaultdict(int)
        self.values = defaultdict(lambda: defaultdict(list))
        self.completed = 0
        for task in tasks:
            for subset in ("all", task.decision):
                self.expected[subset, task.group] += 1

    def add(self, task, scores):
        values = (scores["token_accuracy"]["before"], scores["token_accuracy"]["after"],
                  scores["exact_match"]["before"], scores["exact_match"]["after"],
                  scores["output_token_agreement"], scores["output_exact_agreement"])
        for subset in ("all", task.decision):
            self.values[subset, task.group][task.source_case_id].append(values)
        self.completed += 1

    def export(self):
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
