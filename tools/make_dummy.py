"""BERT 없이 파이프라인 점검용 가짜 processed 데이터셋 생성.

  python tools/make_dummy.py            -> data/processed/dummy/
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main(name="dummy", n_news=300, n_users=200, max_len=5, seed=0):
    rng = np.random.default_rng(seed)
    out = os.path.join(ROOT, "data", "processed", name)
    os.makedirs(out, exist_ok=True)

    labels = (rng.random(n_news) < 0.4).astype(int)
    news = pd.DataFrame({
        "idx": range(1, n_news + 1),
        "news_id": [f"dummy-{i}" for i in range(n_news)],
        "title": [f"title {i}" for i in range(n_news)],
        "description": [f"desc {i}" if i % 3 else "" for i in range(n_news)],
        "label": labels,
    })
    news.to_csv(os.path.join(out, "news.csv"), index=False)

    t_emb = torch.randn(n_news + 1, 768); t_emb[0] = 0
    d_emb = torch.randn(n_news + 1, 768); d_emb[0] = 0
    d_emb[1:][torch.tensor(news["description"].values == "")] = 0
    torch.save({"title": t_emb, "description": d_emb}, os.path.join(out, "news_emb.pt"))

    def tok(L, empty_rows):
        ids = torch.randint(1000, 5000, (n_news + 1, L)); ids[:, 0] = 101
        mask = torch.ones(n_news + 1, L, dtype=torch.long)
        ids[0] = 0; mask[0] = 0
        ids[empty_rows] = 0; mask[empty_rows] = 0
        return ids, mask

    empty = torch.tensor(np.where(news["description"].values == "")[0] + 1)
    t_ids, t_mask = tok(32, torch.tensor([], dtype=torch.long))
    d_ids, d_mask = tok(128, empty)
    torch.save({"title_ids": t_ids, "title_mask": t_mask, "desc_ids": d_ids, "desc_mask": d_mask},
               os.path.join(out, "tokens.pt"))

    inst = []
    for u in range(n_users):
        L = int(rng.integers(2, 12))
        seq = rng.choice(np.arange(1, n_news + 1), size=L, replace=False).tolist()
        t0 = int(rng.integers(1_500_000_000, 1_600_000_000))
        for i in range(1, L):
            inst.append([seq[max(0, i - max_len):i], seq[i], u, t0 + i * 3600])
    inst.sort(key=lambda x: x[3])
    n = len(inst)
    splits = {"train": inst[:int(n * .8)], "val": inst[int(n * .8):int(n * .9)], "test": inst[int(n * .9):]}
    for k, v in splits.items():
        json.dump(v, open(os.path.join(out, f"{k}.json"), "w"))

    meta = {"dataset": name, "num_news": n_news, "num_users": n_users, "max_len": max_len,
            "bert": "none", "emb_dim": 768, "title_len": 32, "desc_len": 128,
            "num_instances": {k: len(v) for k, v in splits.items()}, "unseen": {}, "time_cut": {}}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"-> {out}: news={n_news} users={n_users} inst={ {k: len(v) for k, v in splits.items()} }")


if __name__ == "__main__":
    main()
