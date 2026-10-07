"""전처리 (1회). data/{dataset}.csv -> data/processed/{dataset}/ (임베딩·토큰·카테고리·엔티티·단어·감성·pc·인스턴스 json·meta)
규칙: 뉴스 시각순 앞 20% = P_c(PRISM 분류기용), 뒤 80% 뉴스로 인스턴스(직전 5개 -> 다음 뉴스, 중복 유지) 생성, 무작위 8:1:1 분할
"""

import argparse
import ast
import bz2
import io
import json
import os
import random
import re
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from config import BERT_MODEL, COMMON, DATA_URL, NER_MODEL, SENTIMENT_MODEL, WORD_VEC

HERE = os.path.dirname(os.path.abspath(__file__))
TITLE_LEN = 32
DESC_LEN = 128
SPLIT = (0.8, 0.1, 0.1)
PC_RATIO = 0.2          # 시간순 앞쪽 뉴스 비율 (P_c)
ENTITY_LEN = 5
WORD_TITLE_LEN = 30     # FUM MAX_TITLE
WORD_DESC_LEN = 50      # FUM MAX_CONTENT
WORD_DIM = 300
NUM_CATEGORY = 300      # K-means K 상한. K = min(300, 뉴스 수 // 100)


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


def split_news(df: pd.DataFrame):
    """뉴스 시각(가장 먼저 공유된 시각)순으로 앞 PC_RATIO = P_c. -> Bool [N+1] (True = P_c)"""
    times = []
    for s in df["user_times"]:
        ts = []
        for t in ast.literal_eval(s):
            try:
                ts.append(int(t))
            except (TypeError, ValueError):
                continue
        times.append(min(ts, default=None))
    timed = sorted((i for i, t in enumerate(times) if t is not None), key=lambda i: times[i])
    pc = torch.ones(len(df) + 1, dtype=torch.bool)
    pc[0] = False
    pc[[i + 1 for i in timed[int(len(timed) * PC_RATIO):]]] = False
    return pc


def build_sequences(df: pd.DataFrame):
    """{user: [(time, news_idx)]} 시간순."""
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
        if len(ev) >= 2:
            seqs[u] = ev
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


def random_split(inst: list):
    """인스턴스를 무작위로 섞어 나눈다. -> train, val, test"""
    inst = sorted(inst, key=lambda x: (x[0], x[3]))
    random.Random(COMMON["seed"]).shuffle(inst)
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


def categorize(titles, tokenizer, given=None):
    """제목 TF-IDF -> K-means 카테고리 (given = CSV category 컬럼이 있으면 그 값), 카테고리 안에서 한 번 더 = 서브카테고리.
    -> (category Long [N+1], subcategory Long [N+1])  번호 1.., 0 = 패딩"""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    seed = COMMON["seed"]
    x = TfidfVectorizer(tokenizer=tokenizer.tokenize, lowercase=False, token_pattern=None,
                        max_features=6000).fit_transform(titles)
    if given is None:
        cat = KMeans(min(NUM_CATEGORY, len(titles) // 100), random_state=seed).fit_predict(x)
    else:
        cat = pd.factorize(given)[0]
    sub = np.zeros(len(titles), dtype=np.int64)
    n_sub = 0
    for c in range(cat.max() + 1):
        rows = np.where(cat == c)[0]
        k = max(1, len(rows) // 33)
        sub[rows] = n_sub + KMeans(k, random_state=seed).fit_predict(x[rows])
        n_sub += k

    def pad(a):
        return torch.cat([torch.zeros(1, dtype=torch.long), torch.from_numpy(a).long() + 1])

    return pad(cat), pad(sub)


def extract_entities(titles, ner_name, device, batch=64):
    """제목 -> NER -> 엔티티 번호 (등장 순서대로 1..). -> (Long [N+1, ENTITY_LEN], 엔티티 수)"""
    from transformers import pipeline
    ner = pipeline("token-classification", model=ner_name, aggregation_strategy="first", device=device)
    ner.tokenizer.model_max_length = 512
    ent = torch.zeros(len(titles) + 1, ENTITY_LEN, dtype=torch.long)
    vocab = {}
    for i, found in enumerate(ner(titles, batch_size=batch), start=1):
        names = []
        for e in found:
            name = titles[i - 1][e["start"]:e["end"]].strip().lower()
            if len(name) >= 2 and name not in names:
                names.append(name)
        for j, name in enumerate(names[:ENTITY_LEN]):
            ent[i, j] = vocab.setdefault(name, len(vocab) + 1)
    return ent, len(vocab)


def split_words(text, chinese):
    if chinese:
        import jieba
        return [w for w in jieba.lcut(text) if w.strip()]
    return re.findall(r"\w+|[^\w\s]", text.lower())


def build_words(titles, descs, vec_path, chinese):
    """제목/요약 -> 단어 번호 (등장 순서대로 1..) + 단어 벡터 표. 벡터 파일에 없는 단어는 0 벡터.
    -> ({"title": Long [N+1, 30], "description": Long [N+1, 50], "emb": Float [V+1, 300]}, 벡터 찾은 단어 수)"""
    vocab = {}

    def ids(texts, length):
        out = torch.zeros(len(texts) + 1, length, dtype=torch.long)
        for i, text in enumerate(texts, start=1):
            for j, w in enumerate(split_words(text, chinese)[:length]):
                out[i, j] = vocab.setdefault(w, len(vocab) + 1)
        return out

    words = {"title": ids(titles, WORD_TITLE_LEN), "description": ids(descs, WORD_DESC_LEN)}
    emb = torch.zeros(len(vocab) + 1, WORD_DIM)
    found = 0
    if vec_path.endswith(".zip"):
        z = zipfile.ZipFile(vec_path)
        f = io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8")
    else:
        f = bz2.open(vec_path, "rt", encoding="utf-8", errors="ignore")
    with f:
        for line in f:
            w, _, rest = line.partition(" ")
            if w in vocab:
                v = rest.split()
                if len(v) == WORD_DIM:
                    emb[vocab[w]] = torch.tensor([float(x) for x in v])
                    found += 1
    words["emb"] = emb
    return words, found


def sentiment_scores(titles, model_name, device, batch=64):
    """제목 -> 감성 점수 P(긍정) - P(부정) (RobustSentiRec bert_sentiment 방식). -> Float [N+1]"""
    from transformers import pipeline
    clf = pipeline("text-classification", model=model_name, top_k=None, device=device)
    out = torch.zeros(len(titles) + 1)
    for i, scores in enumerate(clf(titles, batch_size=batch, truncation=True, max_length=512), start=1):
        s = {x["label"].lower()[:3]: x["score"] for x in scores}
        out[i] = s["pos"] - s["neg"]
    return out


def process(dataset: str):
    max_len = COMMON["max_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = os.path.join(HERE, "data", f"{dataset}.csv")
    out = os.path.join(HERE, "data", "processed", dataset)
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(raw):
        urllib.request.urlretrieve(DATA_URL + f"{dataset}.csv", raw)
    print(f"[{dataset}] raw={raw}")

    df = load_raw(raw)
    n_news = len(df)
    print(f"  news={n_news} real={(df.label == 0).sum()} fake={(df.label == 1).sum()} "
          f"no_desc={(df.description == '').sum()}")

    # ---- 뉴스 분리 (P_c / P_u)
    pc = split_news(df)
    torch.save(pc, os.path.join(out, "pc.pt"))
    n_pc = int(pc.sum())
    print(f"  P_c={n_pc} P_u={n_news - n_pc}")

    # ---- 시퀀스(P_u 뉴스만) / 인스턴스 / 분할
    seqs = {u: [e for e in seq if not pc[e[1]]] for u, seq in build_sequences(df).items()}
    seqs = {u: seq for u, seq in seqs.items() if len(seq) >= 2}
    inst = build_instances(seqs, max_len)
    train, val, test = random_split(inst)
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
    bert_name = BERT_MODEL[dataset]
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

    # ---- 카테고리
    cat, sub = categorize(df["title"].tolist(), tok, df.get("category"))
    torch.save(cat, os.path.join(out, "category.pt"))
    torch.save(sub, os.path.join(out, "subcategory.pt"))
    print(f"  category={int(cat.max())} subcategory={int(sub.max())}")

    # ---- 엔티티
    ent, n_ent = extract_entities(df["title"].tolist(), NER_MODEL[dataset], device)
    torch.save(ent, os.path.join(out, "entity.pt"))
    print(f"  entity={n_ent}")

    # ---- 단어 번호 / 단어 벡터
    words, n_found = build_words(df["title"].tolist(), df["description"].tolist(),
                                 os.path.join(HERE, WORD_VEC[dataset]), "chinese" in bert_name)
    torch.save(words, os.path.join(out, "words.pt"))
    print(f"  words={words['emb'].size(0) - 1} (벡터 있음 {n_found})")

    # ---- 감성 점수
    senti = sentiment_scores(df["title"].tolist(), SENTIMENT_MODEL[dataset], device)
    torch.save(senti, os.path.join(out, "sentiment.pt"))
    print(f"  sentiment mean={senti[1:].mean():.3f}")

    meta = {
        "dataset": dataset, "num_news": n_news, "num_users": len(users),
        "num_pc": n_pc, "num_pu": n_news - n_pc, "max_len": max_len, "bert": bert_name,
        "num_category": int(cat.max()), "num_subcategory": int(sub.max()), "num_entity": n_ent,
        "num_instances": {"train": len(train), "val": len(val), "test": len(test)},
        "unseen": {k: v[0] / max(v[1], 1) for k, v in unseen.items()},
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all", choices=list(BERT_MODEL) + ["all"])
    a = p.parse_args()
    for d in (list(BERT_MODEL) if a.dataset == "all" else [a.dataset]):
        process(d)


if __name__ == "__main__":
    main()
