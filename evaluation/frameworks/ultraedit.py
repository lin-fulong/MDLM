"""UltraEdit 问题组和指标；不导入原编辑器或自回归 labels 构造。"""

from evaluation.metrics import paired_scores
from evaluation.types import EvalTask, EvaluationFramework
from util.io import config_digest


class UltraEditFramework(EvaluationFramework):
    # LS 按无关题标准答案评分；编辑前后输出一致率另行记录。
    metrics = {"edit": "ES", "generalization": "GS", "preservation": "LS",
               "persona": "person_score", "portability": "mhop_score"}

    def build_tasks(self, cases, rounds, groups):
        if not isinstance(groups, list) or not groups or len(set(groups)) != len(groups) or set(groups) - self.metrics.keys():
            raise ValueError("groups 必须是不重复的 UltraEdit 问题组列表")
        by_id = {}
        for case in cases:
            if case["case_id"] in by_id:
                raise ValueError(f"重复事实 ID: {case['case_id']}")
            by_id[case["case_id"]] = case
        tasks, seen = [], set()
        # 仅回测编辑报告实际尝试的事实，保留报告和问题组的原顺序。
        for row in rounds:
            for fact_id in row["fact_ids"]:
                if fact_id not in by_id:
                    raise ValueError(f"当前源数据缺少已编辑事实: {fact_id}")
                case = by_id[fact_id]
                if case.get("source_case_id", fact_id) != row["case_id"]:
                    raise ValueError(f"事实来源与编辑报告不符: {fact_id}")
                for group in groups:
                    probes = case["prompts"].get(group, [])
                    if not isinstance(probes, list) or (group == "edit" and not probes):
                        raise ValueError(f"{fact_id}: 非法或缺失 edit 问题组")
                    for probe in probes:
                        if any(not isinstance(probe.get(key), str) or not probe[key].strip()
                               for key in ("probe_id", "prompt", "answer")):
                            raise ValueError(f"{fact_id}/{group}: 探针必须包含非空 ID、问题和答案")
                        task_id = config_digest({"case_id": fact_id, "group": group, "probe_id": probe["probe_id"]})
                        if task_id in seen:
                            raise ValueError(f"重复评估探针: {fact_id}/{probe['probe_id']}")
                        seen.add(task_id)
                        tasks.append(EvalTask(task_id, row["case_id"], fact_id, probe["probe_id"], group,
                                              probe["prompt"], probe["answer"], row["decision"], row["checkpoint"]))
        if not tasks:
            raise ValueError("所选问题组没有可评估任务")
        return tasks

    def score(self, before, after, target):
        return paired_scores(before.token_ids, after.token_ids, target)
