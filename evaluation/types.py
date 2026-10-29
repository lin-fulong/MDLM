"""评估任务与预测；生成请求不携带参考答案。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    source_case_id: str
    case_id: str
    probe_id: str
    group: str
    prompt: str
    answer: str
    decision: str
    checkpoint: Path | None


@dataclass(frozen=True)
class GenerationRequest:
    prompt_ids: tuple[int, ...]
    answer_length: int
    seed: int


@dataclass(frozen=True)
class Prediction:
    token_ids: tuple[int, ...]
    text: str


class EvaluationFramework:
    """自定义框架实现任务构造和评分；公共运行器处理模型和报告。"""

    metrics: dict[str, str]

    def build_tasks(self, cases, rounds, groups):
        raise NotImplementedError

    def score(self, before, after, target):
        raise NotImplementedError

    def aggregate(self, tasks, groups):
        from .reporting import ScoreSummary
        return ScoreSummary(tasks, {group: self.metrics[group] for group in groups})
