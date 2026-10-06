"""PRISM 논문 평가지표, full-ranking (전체 뉴스 라이브러리 순위).

  HR@K    정답이 top-K 안에 있는 비율
  NDCG@K  1/log2(rank+1)
  MRR@K   1/rank (top-K 밖이면 0)
  RT@K    top-K 중 진짜 뉴스 비율
  WFNS@K  1 - sum_{fake in topK}(K - idx) / sum(1..K)   (상위 가짜일수록 큰 패널티)
  1-R     1위가 진짜 뉴스인 비율
"""

import math

import torch


@torch.no_grad()
def evaluate(model, loader, labels: torch.Tensor, ks=(5, 10, 20), device="cpu",
             exclude_history: bool = False) -> dict:
    model.eval()
    labels = labels.to(device)
    K = max(ks)
    cand_mask = model.candidate_mask()
    if cand_mask is not None:
        cand_mask = cand_mask.to(device)

    hit = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    mrr = {k: 0.0 for k in ks}
    real = {k: 0.0 for k in ks}
    wfns = {k: 0.0 for k in ks}
    top1_real = 0.0
    n = 0
    denom = {k: k * (k + 1) / 2 for k in ks}

    for batch in loader:
        batch = model.to_device(batch, device)
        scores = model.score(batch).float()                 # [B, N+1]
        scores[:, 0] = -float("inf")
        if cand_mask is not None:
            scores = scores.masked_fill(cand_mask.unsqueeze(0), -float("inf"))
        if exclude_history:
            ctx = batch["ctx"].masked_fill(~batch["mask"], 0)
            scores.scatter_(1, ctx, -float("inf"))
            scores[:, 0] = -float("inf")

        top = scores.topk(K, dim=1).indices                 # [B, K]
        tgt = batch["target"].unsqueeze(1)                  # [B, 1]
        pos = (top == tgt)                                  # [B, K]
        rank = torch.where(pos.any(1), pos.float().argmax(1) + 1, torch.full_like(tgt.squeeze(1), K + 1))
        fake = labels[top]                                  # [B, K] 1 = fake
        B = top.size(0)
        n += B
        top1_real += (fake[:, 0] == 0).float().sum().item()
        for k in ks:
            inK = rank <= k
            hit[k] += inK.float().sum().item()
            ndcg[k] += (inK.float() / torch.log2(rank.float() + 1)).sum().item()
            mrr[k] += (inK.float() / rank.float()).sum().item()
            fk = fake[:, :k]
            real[k] += (1 - fk).sum().item() / k
            w = torch.arange(k, 0, -1, device=device, dtype=torch.float)   # idx 0 -> k, ..., idx k-1 -> 1
            wfns[k] += (1 - (fk * w).sum(1) / denom[k]).sum().item()

    out = {"n": n}
    for k in ks:
        out[f"HR@{k}"] = hit[k] / n
        out[f"NDCG@{k}"] = ndcg[k] / n
        out[f"MRR@{k}"] = mrr[k] / n
        out[f"RT@{k}"] = real[k] / n
        out[f"WFNS@{k}"] = wfns[k] / n
    out["1-R"] = top1_real / n
    model.train()
    return out


def format_table(m: dict, ks=(5, 10, 20)) -> str:
    rows = ["HR", "NDCG", "MRR", "RT", "WFNS"]
    head = "Metric  " + "".join(f"@{k:<9}" for k in ks)
    lines = [head]
    for r in rows:
        lines.append(f"{r:<8}" + "".join(f"{m[f'{r}@{k}']:<10.4f}" for k in ks))
    lines.append(f"1-R     {m['1-R']:.4f}   (n={m['n']})")
    return "\n".join(lines)
