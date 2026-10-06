"""공통 학습 루프. 모델 종류를 모른다. BaseModel 인터페이스만 사용.

stage 별로: 로더 생성 -> optimizer -> epochs 반복 -> (eval_enabled 이면) val 평가 -> best 유지
stage 끝: best 가중치 복원 -> model.on_stage_end(stage)
"""

import copy
import json
import logging
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from datasets import BenchDataset
from evaluate import evaluate, format_table


class Trainer:
    def __init__(self, model, data: dict, cfg: dict, out_dir: str, logger: logging.Logger):
        self.model = model
        self.data = data
        self.cfg = cfg
        self.out_dir = out_dir
        self.log = logger
        self.device = model.device
        os.makedirs(out_dir, exist_ok=True)
        self.val_loader = self._eval_loader("val")
        self.test_loader = self._eval_loader("test")

    # ---- 로더
    def _eval_loader(self, split):
        ds = BenchDataset(self.data["splits"][split], self.data["labels"],
                          max_len=self.cfg["max_len"], num_neg=self.cfg["num_neg"], train=False)
        return DataLoader(ds, batch_size=self.cfg["eval_batch_size"], shuffle=False,
                          num_workers=self.cfg["num_workers"])

    def _train_loader(self, stage):
        ov = self.model.loader_overrides(stage)
        inst = self.data["splits"]["train"]
        if ov.get("max_samples") and len(inst) > ov["max_samples"]:
            rng = random.Random(self.cfg["seed"] + stage)
            inst = rng.sample(inst, ov["max_samples"])
        ds = BenchDataset(inst, self.data["labels"], max_len=self.cfg["max_len"],
                          num_neg=self.cfg["num_neg"], train=True, seed=self.cfg["seed"] + stage,
                          exclude=self.data["pc"])
        return DataLoader(ds, batch_size=ov.get("batch_size", self.cfg["batch_size"]), shuffle=True,
                          num_workers=self.cfg["num_workers"], drop_last=False)

    def _epochs(self, stage):
        e = self.cfg["epochs"]
        return e[stage] if isinstance(e, (list, tuple)) else int(e)

    # ---- 학습
    def fit(self):
        metric = self.cfg["select_metric"]
        final_best = None
        for stage in range(self.model.num_stages):
            loader = self._train_loader(stage)
            ov = self.model.loader_overrides(stage)
            accum = int(ov.get("grad_accum", 1))
            opt = self.model.configure_optimizer(stage)
            sched = None
            if isinstance(opt, (tuple, list)):
                opt, sched = opt
            self.model.on_stage_start(stage)
            n_ep = self._epochs(stage)
            self.log.info(f"=== stage {stage + 1}/{self.model.num_stages}: epochs={n_ep} "
                          f"batches={len(loader)} batch_size={loader.batch_size} grad_accum={accum}")
            best_val, best_state = None, None
            for ep in range(1, n_ep + 1):
                t0 = time.time()
                self.model.train()
                tot, cnt, agg = 0.0, 0, {}
                opt.zero_grad()
                for i, batch in enumerate(loader):
                    batch = self.model.to_device(batch, self.device)
                    loss, logs = self.model.compute_loss(batch, stage)
                    (loss / accum).backward()
                    if (i + 1) % accum == 0 or (i + 1) == len(loader):
                        opt.step()
                        opt.zero_grad()
                    tot += loss.item()
                    cnt += 1
                    for k, v in logs.items():
                        agg[k] = agg.get(k, 0.0) + float(v)
                if sched is not None:
                    sched.step()
                msg = f"[stage {stage + 1}] epoch {ep}/{n_ep} loss={tot / max(cnt, 1):.4f}"
                if agg:
                    msg += " " + " ".join(f"{k}={v / max(cnt, 1):.4f}" for k, v in agg.items())
                msg += f" ({time.time() - t0:.0f}s)"
                if self.model.eval_enabled(stage) and (ep % self.cfg["eval_every"] == 0 or ep == n_ep):
                    m = evaluate(self.model, self.val_loader, self.data["labels"], self.cfg["ks"],
                                 self.device, self.cfg["exclude_history"])
                    msg += f" | val {metric}={m[metric]:.4f} RT@5={m['RT@5']:.4f}"
                    if best_val is None or m[metric] > best_val:
                        best_val = m[metric]
                        best_state = copy.deepcopy(self.model.state_dict())
                        msg += " *"
                self.log.info(msg)
            if best_state is not None:
                self.model.load_state_dict(best_state)
                self.log.info(f"stage {stage + 1} best val {metric}={best_val:.4f} restored")
                final_best = best_val
            self.model.on_stage_end(stage)
        torch.save({"state_dict": self.model.state_dict(), "cfg": self.cfg,
                    "val_metric": final_best}, os.path.join(self.out_dir, "best.pt"))
        self.log.info(f"saved {os.path.join(self.out_dir, 'best.pt')}")

    # ---- 평가
    def test(self) -> dict:
        path = os.path.join(self.out_dir, "best.pt")
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["state_dict"])
        m = evaluate(self.model, self.test_loader, self.data["labels"], self.cfg["ks"],
                     self.device, self.cfg["exclude_history"])
        self.log.info("test\n" + format_table(m, self.cfg["ks"]))
        with open(os.path.join(self.out_dir, "test_results.json"), "w") as f:
            json.dump(m, f, indent=2)
        return m
