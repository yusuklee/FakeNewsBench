"""모델 레지스트리. main.py --model 이름 -> 클래스."""
import importlib

REGISTRY = {
    "meanpool": ("models.meanpool", "MeanPool"),
    "rec4mit": ("models.rec4mit", "Rec4Mit"),
    "hdint": ("models.hdint", "HDInt"),
    "prism": ("models.prism", "PRISM"),
}


def build_model(name: str, data: dict, cfg: dict, device):
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {list(REGISTRY)}")
    mod, cls = REGISTRY[name]
    return getattr(importlib.import_module(mod), cls)(data, cfg, device)
