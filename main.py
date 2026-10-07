"""학습 + 평가 진입점.  python main.py --model prism --dataset gossip [--test_only]
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import torch

from config import BERT_MODEL, build_config
from datasets import load_processed
from models import REGISTRY, build_model
from trainer import Trainer

HERE = os.path.dirname(os.path.abspath(__file__))


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
    p.add_argument("--dataset", required=True, choices=list(BERT_MODEL))
    p.add_argument("--test_only", action="store_true")
    a = p.parse_args()

    cfg = build_config(a.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(HERE, "checkpoints", a.model, a.dataset)
    log = get_logger(os.path.join(HERE, "logs", a.model, a.dataset, datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"))
    log.info(f"model={a.model} dataset={a.dataset} device={device}")
    log.info("cfg " + " ".join(f"{k}={v}" for k, v in cfg.items()))

    data = load_processed(a.dataset)
    meta = data["meta"]
    log.info(f"news={data['num_news']} users={data['num_users']} "
             f"inst train/val/test={meta['num_instances']['train']}/{meta['num_instances']['val']}/{meta['num_instances']['test']}")
    if meta["max_len"] < cfg["max_len"]:
        log.warning(f"processed max_len={meta['max_len']} < cfg max_len={cfg['max_len']}")

    model = build_model(a.model, data, cfg, device).to(device)
    log.info(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    tr = Trainer(model, data, cfg, out_dir, log)
    if not a.test_only:
        tr.fit()
    tr.test()


if __name__ == "__main__":
    main()
