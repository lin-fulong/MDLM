"""设置单进程算法运行所用的设备与 TF32 精度。"""

from contextlib import contextmanager
import torch


@contextmanager
def execution_context(settings):
    """选择计算设备；resident 模式关闭 TF32，并在退出时恢复原设置。

    本函数不实现公式（1）～（18），只固定这些计算共同使用的运行环境。
    """
    backend = settings.get("backend", "subprocess")
    if backend not in ("resident", "subprocess"):
        raise ValueError("execution.backend must be resident or subprocess")
    if "device" in settings or backend == "resident":
        device = torch.device(settings.get("device", "cuda:0"))
        if device.type not in ("cpu", "cuda"):
            raise ValueError("execution.device must be cpu or cuda:<index>")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("当前 Python 无法使用 CUDA；请检查 mdlm 环境或显式使用 --device cpu")
            torch.cuda.set_device(device)
            print(f"[GPU] {torch.cuda.get_device_name(device)} | {device}", flush=True)
    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        if backend == "resident":
            # 沿用原驻留流程的算术精度，避免入口重构改变 float32 数值。
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
