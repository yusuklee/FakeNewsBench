"""학습 + 평가 진입점.

  python main.py --model prism   --dataset gossip
  python main.py --model hdint   --dataset pheme --set epochs=[1,2]
  python main.py --model rec4mit --dataset gossip --test_only
"""

import argparse
import ast
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

from config import RAW_FILES, build_config
from datasets import load_processed
from models import REGISTRY, build_model
from trainer import Trainer

HERE = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_overrides(items):
    out = {}
    for s in items or []:
        k, _, v = s.partition("=")
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def get_logger(path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lg = logging.getLogger("bench")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(REGISTRY))
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_root", default=os.path.join(HERE, "data", "processed"))
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--ckpt", default=None, help="test_only 시 체크포인트 경로 (기본 checkpoints/{model}/{dataset}/best.pt)")
    p.add_argument("--tag", default="", help="checkpoints/{model}/{dataset}{tag}/ 로 저장")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--set", nargs="*", default=[], help="config 덮어쓰기 key=value (예: epochs=[1,2] lr=0.0005)")
    a = p.parse_args()

    cfg = build_config(a.model, parse_overrides(a.set))
    if a.seed is not None:
        cfg["seed"] = a.seed
    set_seed(cfg["seed"])

    run = f"{a.dataset}{a.tag}"
    out_dir = os.path.join(HERE, "checkpoints", a.model, run)
    log = get_logger(os.path.join(HERE, "logs", a.model, run, datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"))
    log.info(f"model={a.model} dataset={a.dataset} device={a.device}")
    log.info("cfg " + " ".join(f"{k}={v}" for k, v in cfg.items()))

    data = load_processed(a.dataset, a.data_root)
    meta = data["meta"]
    log.info(f"news={data['num_news']} users={data['num_users']} "
             f"inst train/val/test={meta['num_instances']['train']}/{meta['num_instances']['val']}/{meta['num_instances']['test']}")
    if meta["max_len"] < cfg["max_len"]:
        log.warning(f"processed max_len={meta['max_len']} < cfg max_len={cfg['max_len']}")

    device = torch.device(a.device)
    model = build_model(a.model, data, cfg, device).to(device)
    log.info(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    tr = Trainer(model, data, cfg, out_dir, log)
    if not a.test_only:
        tr.fit()
    tr.test(a.ckpt)


if __name__ == "__main__":
    main()
