"""按名称选择框架；新框架继承 EvaluationFramework 并在这里注册。"""

from .frameworks.ultraedit import UltraEditFramework

FRAMEWORKS = {"ultraedit": UltraEditFramework}


def get_framework(name):
    if name not in FRAMEWORKS:
        raise ValueError(f"未注册的评估框架: {name}")
    return FRAMEWORKS[name]()
