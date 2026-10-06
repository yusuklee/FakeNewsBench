"""파이프라인 점검용 최소 베이스라인.

히스토리 임베딩 평균 -> 선형 투영 -> 전체 뉴스와 코사인. 학습: 정답 + 네거티브 CE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


class MeanPool(BaseModel):
    def __init__(self, data, cfg, device):
        super().__init__(data, cfg, device)
        emb = self.full_emb()                                   # [N+1, 1536]
        self.register_buffer("news_emb", emb)
        self.proj = nn.Linear(emb.size(1), cfg.get("dim", 256))
        self.scale = cfg.get("scale", 10.0)

    def _user(self, batch):
        e = self.news_emb[batch["ctx"]]                          # [B, L, D]
        m = batch["mask"].unsqueeze(-1).float()
        u = (e * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.proj(u), dim=-1)

    def compute_loss(self, batch, stage=0):
        u = self._user(batch)                                    # [B, d]
        cand = torch.cat([batch["target"].unsqueeze(1), batch["neg"]], 1)   # [B, 1+K]
        c = F.normalize(self.proj(self.news_emb[cand]), dim=-1)  # [B, 1+K, d]
        logits = (u.unsqueeze(1) * c).sum(-1) * self.scale
        loss = F.cross_entropy(logits, torch.zeros(len(u), dtype=torch.long, device=u.device))
        return loss, {}

    @torch.no_grad()
    def score(self, batch):
        u = self._user(batch)
        items = F.normalize(self.proj(self.news_emb), dim=-1)    # [N+1, d]
        return u @ items.T
