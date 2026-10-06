"""전처리 (1회 실행). 모든 모델이 같은 산출물을 쓴다.

  python prepare_data.py --dataset gossip        # gossip / pol / pheme / ced / mcfend / all

입력  : data/{dataset}.csv  (없으면 GitHub release v0.1.0 에서 받아 저장)
        컬럼 news_id, title, description, label(0 real / 1 fake), user_ids, user_times
산출  : data/processed/{dataset}/
        news_emb.pt   {"title": [N+1,768], "description": [N+1,768]}  BERT CLS, row 0 = 패딩
        tokens.pt     {"title_ids","title_mask":[N+1,32], "desc_ids","desc_mask":[N+1,128]}
        train.json / val.json / test.json   [[ctx_ids], target_id, user_idx, target_time]
        meta.json

규칙
  - description이 비어 있거나 title과 같으면 description은 없는 것으로 처리 (임베딩/토큰 0)
  - 유저 시퀀스: user_times 기준 시간순, 같은 뉴스 재등장은 첫 번째만 유지
  - 인스턴스: 슬라이딩 윈도우, context = 직전 최대 max_len개, target = 다음 뉴스
  - 분할: 인스턴스 전체를 target_time 순으로 정렬 → 앞 80% train / 10% val / 10% test
"""

import argparse
import ast
import json
import os
import urllib.request
from collections import defaultdict

import pandas as pd
import torch

from config import BERT_BY_DATASET, RAW_FILES, RAW_URL

HERE = os.path.dirname(os.path.abspath(__file__))
TITLE_LEN = 32
DESC_LEN = 128
SPLIT = (0.8, 0.1, 0.1)


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["news_id"] = df["news_id"].astype(str)
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    same = df["description"] == df["title"]
    df.loc[same, "description"] = ""
    df.loc[df["title"] == "", "title"] = "unknown news"
    return df.reset_index(drop=True)


def build_sequences(df: pd.DataFrame):
    """{user: [(time, news_idx)]} 시간순, 중복 뉴스는 첫 등장만."""
    events = defaultdict(list)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        users = ast.literal_eval(row.user_ids)
        times = ast.literal_eval(row.user_times)
        for u, t in zip(users, times):
            try:
                t = int(t)
            except (TypeError, ValueError):
                continue
            events[str(u)].append((t, idx))
    seqs = {}
    for u, ev in events.items():
        ev.sort(key=lambda x: x[0])
        seen, out = set(), []
        for t, n in ev:
            if n not in seen:
                seen.add(n)
                out.append((t, n))
        if len(out) >= 2:
            seqs[u] = out
    return seqs


def build_instances(seqs: dict, max_len: int):
    """슬라이딩 윈도우 -> [(target_time, ctx, target, user_str)]"""
    inst = []
    for u, seq in seqs.items():
        for i in range(1, len(seq)):
            ctx = [n for _, n in seq[max(0, i - max_len):i]]
            t, tgt = seq[i]
            inst.append((t, ctx, tgt, u))
    return inst


def temporal_split(inst: list):
    inst = sorted(inst, key=lambda x: (x[0], x[3]))
    n = len(inst)
    n_tr = int(n * SPLIT[0])
    n_va = int(n * (SPLIT[0] + SPLIT[1]))
    return inst[:n_tr], inst[n_tr:n_va], inst[n_va:]


@torch.no_grad()
def encode(texts, tokenizer, model, max_len, device, batch=128):
    """BERT CLS 임베딩 + 토큰. 빈 문자열은 0."""
    n = len(texts)
    emb = torch.zeros(n + 1, model.config.hidden_size)
    ids = torch.zeros(n + 1, max_len, dtype=torch.long)
    mask = torch.zeros(n + 1, max_len, dtype=torch.long)
    nonempty = [i for i, t in enumerate(texts) if t]
    for s in range(0, len(nonempty), batch):
        rows = nonempty[s:s + batch]
        enc = tokenizer([texts[i] for i in rows], padding="max_length", truncation=True,
                        max_length=max_len, return_tensors="pt")
        out = model(**{k: v.to(device) for k, v in enc.items()})
        cls = out.last_hidden_state[:, 0, :].float().cpu()
        for j, i in enumerate(rows):
            emb[i + 1] = cls[j]
            ids[i + 1] = enc["input_ids"][j]
            mask[i + 1] = enc["attention_mask"][j]
        print(f"    {min(s + batch, len(nonempty))}/{len(nonempty)}", end="\r")
    print()
    return emb, ids, mask


def process(dataset: str, out_dir: str, max_len: int, device: str):
    raw = os.path.join(HERE, "data", RAW_FILES[dataset])
    out = os.path.join(out_dir, dataset)
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(raw):
        urllib.request.urlretrieve(RAW_URL + RAW_FILES[dataset], raw)
    print(f"[{dataset}] raw={raw}")

    df = load_raw(raw)
    n_news = len(df)
    print(f"  news={n_news} real={(df.label == 0).sum()} fake={(df.label == 1).sum()} "
          f"no_desc={(df.description == '').sum()}")

    # ---- 시퀀스 / 인스턴스 / 분할
    seqs = build_sequences(df)
    inst = build_instances(seqs, max_len)
    train, val, test = temporal_split(inst)
    users = sorted({u for _, _, _, u in inst}, key=str)
    u2i = {u: i for i, u in enumerate(users)}
    print(f"  users={len(users)} instances={len(inst)} "
          f"train/val/test={len(train)}/{len(val)}/{len(test)}")

    def dump(split, name):
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump([[ctx, tgt, u2i[u], t] for t, ctx, tgt, u in split], f)

    dump(train, "train")
    dump(val, "val")
    dump(test, "test")

    def lib(s):
        return {n for _, ctx, tgt, _ in s for n in ctx + [tgt]}

    T = lib(train)
    unseen = {name: (len(lib(s) - T), len(lib(s))) for name, s in [("val", val), ("test", test)]}
    print(f"  unseen val={unseen['val'][0]}/{unseen['val'][1]} test={unseen['test'][0]}/{unseen['test'][1]}")

    # ---- BERT 임베딩 / 토큰
    from transformers import AutoModel, AutoTokenizer
    bert_name = BERT_BY_DATASET[dataset]
    print(f"  BERT={bert_name} device={device}")
    tok = AutoTokenizer.from_pretrained(bert_name)
    bert = AutoModel.from_pretrained(bert_name).to(device).eval()
    print("  title")
    t_emb, t_ids, t_mask = encode(df["title"].tolist(), tok, bert, TITLE_LEN, device)
    print("  description")
    d_emb, d_ids, d_mask = encode(df["description"].tolist(), tok, bert, DESC_LEN, device)
    torch.save({"title": t_emb, "description": d_emb}, os.path.join(out, "news_emb.pt"))
    torch.save({"title_ids": t_ids, "title_mask": t_mask,
                "desc_ids": d_ids, "desc_mask": d_mask}, os.path.join(out, "tokens.pt"))

    meta = {
        "dataset": dataset, "num_news": n_news, "num_users": len(users),
        "max_len": max_len, "bert": bert_name, "emb_dim": int(t_emb.shape[1]),
        "title_len": TITLE_LEN, "desc_len": DESC_LEN,
        "num_instances": {"train": len(train), "val": len(val), "test": len(test)},
        "unseen": {k: v[0] / max(v[1], 1) for k, v in unseen.items()},
        "time_cut": {"train_end": train[-1][0] if train else None,
                     "val_end": val[-1][0] if val else None},
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all", choices=list(RAW_FILES) + ["all"])
    p.add_argument("--out_dir", default=os.path.join(HERE, "data", "processed"))
    p.add_argument("--max_len", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    targets = list(RAW_FILES) if a.dataset == "all" else [a.dataset]
    for d in targets:
        process(d, a.out_dir, a.max_len, a.device)


if __name__ == "__main__":
    main()
