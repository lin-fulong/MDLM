"""评价任务、UltraEdit 题组及轻量框架注册；自定义框架只需提供同名方法。"""

from dataclasses import dataclass
from pathlib import Path

from .reporting import ScoreSummary, paired_scores
from util.io import config_digest


@dataclass(frozen=True)
class EvalTask:
    """一道评价题；同时保留源案例、事实、题组和 after 权重路径。"""

    task_id: str
    source_case_id: str
    case_id: str
    probe_id: str
    group: str
    prompt: str
    answer: str
    decision: str
    checkpoint: Path | None


class UltraEditFramework:
    """五类题目共用 token 正确率，区别仅在问题与标准答案来源。"""

    # LS 按无关题标准答案评分；编辑前后输出一致率另行记录。
    metrics = {"edit": "ES", "generalization": "GS", "preservation": "LS",
               "persona": "person_score", "portability": "mhop_score"}

    def build_tasks(self, cases, rounds, groups):
        """仅展开编辑报告实际尝试的事实，保持源案例和题目顺序。"""
        if not groups or len(set(groups)) != len(groups) or set(groups) - self.metrics.keys():
            raise ValueError("groups 必须是不重复的 UltraEdit 问题组列表")
        by_id = {case["case_id"]: case for case in cases}
        if len(by_id) != len(cases):
            raise ValueError("源数据包含重复事实 ID")
        tasks, seen = [], set()
        # 仅回测编辑报告实际尝试的事实，保留报告和问题组的原顺序。
        for row in rounds:
            for fact_id in row["fact_ids"]:
                case = by_id[fact_id]
                if case.get("source_case_id", fact_id) != row["case_id"]:
                    raise ValueError(f"事实来源与编辑报告不符: {fact_id}")
                for group in groups:
                    probes = case["prompts"].get(group, [])
                    if group == "edit" and not probes:
                        raise ValueError(f"{fact_id}: 缺失 edit 问题组")
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
        """s = Σ 1[预测 token = 标准 token] / L；回填状态不参与评分。"""
        return paired_scores(before.token_ids, after.token_ids, target)

    def aggregate(self, tasks, groups):
        """建立按题目、按源案例两种平均的统计器。"""
        return ScoreSummary(tasks, {group: self.metrics[group] for group in groups})


FRAMEWORKS = {"ultraedit": UltraEditFramework}


def get_framework(name):
    """按配置名称创建框架；后续框架只需提供相同方法。"""
    return FRAMEWORKS[name]()
