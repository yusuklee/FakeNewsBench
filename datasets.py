"""processed/ 로드 + 인스턴스를 배치 텐서로 변환하는 Dataset (모델 공통, 분기 없음).

배치 키
  ctx          Long [L]   히스토리 뉴스 idx, 오른쪽 0 패딩
  mask         Bool [L]   True = 실제 뉴스
  target       Long       정답 뉴스 idx
  user         Long       유저 idx
  neg          Long [K]   네거티브 (real K/2 + fake K/2, target 제외). 평가 시 0
  ctx_label    Float [L]  히스토리 진위 (0 real / 1 fake, 패딩 0)
  target_label Float
  neg_label    Float [K]
임베딩/토큰 조회는 모델이 한다 (data["emb"], data["tokens"]).
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))


def load_processed(dataset: str, root: str | None = None, need_tokens: bool = True) -> dict:
    d = os.path.join(root or os.path.join(HERE, "data", "processed"), dataset)
    meta = json.load(open(os.path.join(d, "meta.json")))
    news = pd.read_csv(os.path.join(HERE, "data", f"{dataset}.csv"), usecols=["label"])
    n = meta["num_news"]
    labels = torch.zeros(n + 1, dtype=torch.float)
    labels[1:] = torch.tensor(news["label"].values, dtype=torch.float)
    emb = torch.load(os.path.join(d, "news_emb.pt"))
    out = {
        "name": dataset, "dir": d, "meta": meta,
        "num_news": n, "num_users": meta["num_users"],
        "labels": labels,                      # [N+1] 0 real / 1 fake
        "emb": emb,                            # {"title": [N+1,768], "description": [N+1,768]}
        "tokens": None,
        "splits": {s: json.load(open(os.path.join(d, f"{s}.json"))) for s in ("train", "val", "test")},
    }
    if need_tokens and os.path.exists(os.path.join(d, "tokens.pt")):
        out["tokens"] = torch.load(os.path.join(d, "tokens.pt"))
    return out


class BenchDataset(Dataset):
    def __init__(self, instances: list, labels: torch.Tensor, max_len: int = 5,
                 num_neg: int = 4, train: bool = True, seed: int = 42):
        self.inst = instances
        self.labels = labels
        self.max_len = max_len
        self.num_neg = num_neg
        self.train = train
        lab = labels[1:].numpy()
        self.real_pool = np.where(lab == 0)[0] + 1
        self.fake_pool = np.where(lab == 1)[0] + 1
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.inst)

    def _sample_neg(self, target: int) -> list:
        k = self.num_neg
        if k == 0:
            return []
        half = k // 2
        out = []
        for pool, m in ((self.real_pool, half), (self.fake_pool, k - half)):
            if len(pool) == 0:
                continue
            tries = 0
            while m > 0 and tries < 100:
                c = int(self.rng.choice(pool))
                tries += 1
                if c != target and c not in out:
                    out.append(c)
                    m -= 1
        while len(out) < k:  # 한쪽 풀이 비어 있을 때 보충
            c = int(self.rng.integers(1, len(self.labels)))
            if c != target and c not in out:
                out.append(c)
        return out

    def __getitem__(self, i):
        ctx, tgt, user, _ = self.inst[i]
        ctx = ctx[-self.max_len:]
        L = self.max_len
        pad = L - len(ctx)
        ctx_t = torch.tensor(ctx + [0] * pad, dtype=torch.long)
        mask = torch.tensor([True] * len(ctx) + [False] * pad)
        neg = self._sample_neg(tgt) if self.train else [0] * self.num_neg
        neg_t = torch.tensor(neg, dtype=torch.long)
        return {
            "ctx": ctx_t,
            "mask": mask,
            "target": torch.tensor(tgt, dtype=torch.long),
            "user": torch.tensor(user, dtype=torch.long),
            "neg": neg_t,
            "ctx_label": self.labels[ctx_t] * mask,
            "target_label": self.labels[tgt],
            "neg_label": self.labels[neg_t],
        }
