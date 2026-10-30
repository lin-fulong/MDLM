"""评估专用的定长全 mask 推理；不使用编辑轨迹、取键或优化代码。"""

import hashlib
from dataclasses import dataclass

import torch
from safetensors.torch import load_file

from model.base import KeepModelInput
from model.registry import get_adapter


PROTOCOL = "diffusion_short_v1"
GOLD_PROTOCOL = "diffusion_gold_v1"
PROTOCOLS = {
    PROTOCOL: "参考答案仅决定窗口长度；全 mask 自主去噪；不是原自回归 teacher forcing",
    GOLD_PROTOCOL: "按置信度选位；先保存预测并评分，再统一回填标准 token；两侧分别选位",
}


@dataclass(frozen=True)
class GenerationRequest:
    """不可变的提示、答案长度和任务种子；不包含标准答案内容。"""

    prompt_ids: tuple[int, ...]
    answer_length: int
    seed: int


@dataclass(frozen=True)
class Prediction:
    """评分用 token 和文本；回填协议另存逐位置对错及逐轮轨迹。"""

    # 新协议的 token_ids 是评分预测，绝不返回回填后的 state 用于评分。
    token_ids: tuple[int, ...]
    text: str
    correct: tuple[int, ...] | None = None
    trace: tuple[dict, ...] = ()


def prepare_request(adapter, task, seed, protocol=PROTOCOL):
    """p = Tok(Render(q))，y = Tok(Render(q) + answer)[len(p):]。"""
    # 联合分词后按提示长度取答案；数据与 tokenizer 需满足稳定的答案边界。
    rendered = adapter.render_prompt(task.prompt)
    prompt = tuple(adapter.tokenizer.encode(rendered, add_special_tokens=False))
    full = tuple(adapter.tokenizer.encode(rendered + task.answer, add_special_tokens=False))
    target = full[len(prompt):]
    derived = int.from_bytes(hashlib.sha256(f"{seed}:{protocol}:{task.task_id}".encode()).digest()[:8], "big") % (2**63)
    # 请求只含提示、长度和种子；标准答案由运行器按协议单独传递。
    return GenerationRequest(prompt, len(target), derived), target


@torch.no_grad()
def generate(model, adapter, request, *, steps=None, temperature=0.0, protocol=PROTOCOL, target=None):
    """从全 mask 窗口去噪；按协议写预测或先评分后回填，返回评分预测。"""
    gold = protocol == GOLD_PROTOCOL
    length = request.answer_length
    blocked = set(adapter.tokenizer.all_special_ids) | {adapter.mask_token_id}
    if gold:
        target = tuple(target) if target is not None else ()
    begin = len(request.prompt_ids)
    # K = L（steps=null），否则 K = min(steps, L)；state 初态为 [p, MASK × L]。
    steps = min(steps, length) if steps is not None else length
    state = list(request.prompt_ids) + [adapter.mask_token_id] * length
    remaining = list(range(begin, len(state)))
    scored_predictions = [None] * length if gold else []
    correct, trace = [None] * length if gold else [], []
    generator = None
    for step in range(steps):
        current = KeepModelInput(tuple(state), (1,) * len(state), tuple(remaining))
        logits = adapter.forward_logits(model, current, tuple(remaining)).to(dtype=torch.float32, copy=True)
        logits[:, [i for i in blocked if 0 <= i < logits.shape[-1]]] = -torch.inf
        # P = softmax(logits / T)；T=0 时用 softmax(logits) 计算置信度，预测取 argmax。
        probabilities = torch.softmax(logits / temperature if temperature else logits, dim=-1)
        if temperature:
            if generator is None:
                # 每道题独立随机流，不推进全局 RNG。
                generator = torch.Generator(device=logits.device).manual_seed(request.seed)
            predicted = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        else:
            predicted = logits.argmax(-1)
        # h_i = P_i(候选 token)，选位不使用标准 token 的概率或预测对错。
        confidence = probabilities.gather(1, predicted[:, None]).squeeze(-1)
        # 每轮统一转到 CPU，避免排序和记录时逐 token 同步 GPU。
        predicted, confidence = predicted.tolist(), confidence.tolist()
        # c_t = floor(L/K) + 1[t <= L mod K]；代码中的 step 从 0 开始。
        count = length // steps + (step < length % steps)
        # S_t 取 h_i 最高的 c_t 个位置，并列时按原位置升序。
        selected = sorted(range(len(remaining)), key=lambda i: (-confidence[i], remaining[i]))[:count]
        # 先保存本轮全部评分；任何位置回填前，所有预测均来自同一次前向。
        if gold:
            positions = [remaining[i] - begin for i in selected]
            for i, position in zip(selected, positions):
                scored_predictions[position] = predicted[i]
                correct[position] = int(scored_predictions[position] == target[position])  # r_i = 1[ŷ_i = y_i]
            trace.append({
                "step": step + 1, "positions": positions,
                "predicted_ids": [scored_predictions[p] for p in positions],
                "confidence": [confidence[i] for i in selected],
                "target_ids": [target[p] for p in positions],
                "correct": [correct[p] for p in positions],
            })
        # 再统一回填：新协议 x_i ← y_i；旧协议 x_i ← ŷ_i，仅修改本轮 S_t。
        for i in selected:
            state[remaining[i]] = target[remaining[i] - begin] if gold else predicted[i]
        chosen = set(selected)
        remaining = [position for i, position in enumerate(remaining) if i not in chosen]  # R ← R \ S_t
    ids = tuple(scored_predictions if gold else state[begin:])
    return Prediction(ids, adapter.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                      tuple(correct) if gold else None, tuple(trace))


class LLaDABackend:
    """只操作本评估进程的模型副本，切换独立案例前恢复原始目标权重。"""

    def __init__(self, settings, device):
        """加载独立评价模型，保存 θ_0 的目标层快照。"""
        self.adapter = get_adapter(model_path=settings["model_path"], target_layers=tuple(settings["target_layers"]))
        self.model = self.adapter.load_model(settings["model_path"], torch_dtype=settings["torch_dtype"], device_map=device)
        self.original = self.adapter.snapshot_target_weights(self.model)

    def use_checkpoint(self, path):
        """先恢复 W_0，再覆盖保存的最终目标权重；path=None 时仅恢复。"""
        self.adapter.restore_target_weights(self.model, self.original)
        if path is None:
            return
        weights = load_file(str(path / "target_weights.safetensors"))
        names = self.adapter.target_module_names()
        parameters = {name: self.model.get_submodule(name).weight for name in names}
        values = {name: weights[name].to(parameter.device) for name, parameter in parameters.items()}
        loaded = False
        try:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(values[name])
            loaded = True
        finally:
            # 张量复制交给 PyTorch；中途失败时恢复，防止留下半加载的模型。
            if not loaded:
                self.adapter.restore_target_weights(self.model, self.original)

    def close(self):
        """恢复原始目标层，并释放后端持有的模型和快照引用。"""
        self.adapter.restore_target_weights(self.model, self.original)
        del self.model, self.original
