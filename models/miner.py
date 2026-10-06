"""MINER (ACL'22 Findings). 1단계: BERT fine-tune + 다중 관심사.

생략: 카테고리 임베딩의 GloVe 초기화 (카테고리가 이름 없는 번호라 처음부터 학습), 학습률 워밍업/선형 감쇠.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from models.base import BaseModel


class PolyAttention(nn.Module):
    """컨텍스트 코드 K개로 히스토리를 K번 가산 어텐션 -> 관심사 벡터 K개. bias = 카테고리 유사도."""

    def __init__(self, dim, num_codes, code_dim):
        super().__init__()
        self.linear = nn.Linear(dim, code_dim, bias=False)
        self.codes = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_codes, code_dim),
                                                          gain=nn.init.calculate_gain("tanh")))

    def forward(self, h, mask, bias):  # h [B, L, D], mask [B, L], bias [B, C, L] -> [B, C, K, D]
        w = (torch.tanh(self.linear(h)) @ self.codes.T).transpose(1, 2)           # [B, K, L]
        w = w.unsqueeze(1) + bias.unsqueeze(2)                                    # [B, C, K, L]
        w = F.softmax(w.masked_fill(~mask[:, None, None, :], -1e9), dim=-1)
        return w @ h.unsqueeze(1)


class MINER(BaseModel):
    def __init__(self, data, cfg, device):
        super().__init__(data, cfg, device)
        tok = data["tokens"]
        ids, att = tok["title_ids"].clone(), tok["title_mask"].clone()
        ids[0, 0], att[0, 0] = ids[1, 0], 1                                        # 패딩 뉴스(0번)도 [CLS] 하나는 넣는다
        self.register_buffer("ids", ids, persistent=False)                         # [N+1, 32]
        self.register_buffer("att", att, persistent=False)
        self.register_buffer("category", self.load_file("category.pt"), persistent=False)  # [N+1]
        self.bert = AutoModel.from_pretrained(data["meta"]["bert"])
        dim = self.bert.config.hidden_size
        self.drop = nn.Dropout(cfg["dropout"])
        self.cat_emb = nn.Embedding(data["meta"]["num_category"] + 1, cfg["category_dim"], padding_idx=0)
        self.lam = nn.Parameter(torch.ones(()))                                    # 논문 Eq 5 의 λ (학습되는 스칼라)
        self.poly = PolyAttention(dim, cfg["num_codes"], cfg["code_dim"])
        self.target_fc = nn.Linear(dim, dim, bias=False)

    def news_vec(self, idx):  # [n] -> [n, D]  BERT [CLS]
        return self.bert(input_ids=self.ids[idx], attention_mask=self.att[idx]).last_hidden_state[:, 0]

    def cat_sim(self, cand_cat, ctx):  # cand_cat [B, C] 또는 [C], ctx [B, L] -> λ·cos [B, C, L]
        a = F.normalize(self.cat_emb(cand_cat), dim=-1)
        b = F.normalize(self.cat_emb(self.category[ctx]), dim=-1)
        return self.lam * (a @ b.transpose(1, 2))

    def match(self, cand, e, eq):
        """후보 벡터와 관심사 벡터 K개 -> 점수 (관심사별 점수의 target-aware 가중합). eq = einsum 식"""
        s = torch.einsum(eq, e, cand)                                              # [B, C, K] 관심사별 점수
        w = F.softmax(torch.einsum(eq, e, F.gelu(self.target_fc(cand))), dim=-1)
        return (w * s).sum(-1)

    # ---- 인터페이스
    def configure_optimizer(self, stage=0):
        return torch.optim.Adam(self.parameters(), lr=self.cfg["lr"])

    def loader_overrides(self, stage):
        return {"batch_size": self.cfg["batch_size"], "grad_accum": self.cfg["grad_accum"],
                "max_samples": self.cfg["max_samples"]}

    def compute_loss(self, batch, stage=0):
        ctx, mask = batch["ctx"], batch["mask"]
        cand = torch.cat([batch["target"].unsqueeze(1), batch["neg"]], 1)          # [B, 1+K], 0번이 정답
        uniq, inv = torch.unique(torch.cat([ctx, cand], 1), return_inverse=True)   # 배치 안 같은 뉴스는 BERT 한 번만
        vec = self.drop(self.news_vec(uniq))[inv]                                  # [B, L+1+K, D]
        h, c = vec[:, :ctx.size(1)], vec[:, ctx.size(1):]
        e = self.poly(h, mask, self.cat_sim(self.category[cand], ctx))             # [B, C, K, D]
        logits = self.match(c, e, "bckd,bcd->bck")
        loss_rec = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))
        # 불일치 정규화: 관심사 벡터끼리의 코사인 유사도를 낮춘다 (Eq 3)
        en = F.normalize(e, dim=-1)
        sim = en @ en.transpose(-1, -2)                                            # [B, C, K, K]
        K = sim.size(-1)
        loss_d = (sim.sum((-1, -2)) - sim.diagonal(dim1=-2, dim2=-1).sum(-1)).mean() / (K * K)
        loss = loss_rec + self.cfg["beta"] * loss_d
        return loss, {"L_rec": loss_rec.item(), "L_d": loss_d.item()}

    @torch.no_grad()
    def score(self, batch):
        news = self.all_news_vecs(self.news_vec, chunk=256)                        # [N+1, D]
        ctx, mask = batch["ctx"], batch["mask"]
        h = news[ctx]
        out = torch.zeros(len(ctx), len(news), device=news.device)
        if "by_cat" not in self._cache:                                            # 카테고리별 뉴스 목록
            self._cache["by_cat"] = [(c, (self.category == c).nonzero().squeeze(1)) for c in self.category.unique()]
        for c, idx in self._cache["by_cat"]:                                       # 관심사 벡터가 후보 카테고리에 따라 달라짐
            e = self.poly(h, mask, self.cat_sim(c.view(1), ctx)).squeeze(1)        # [B, K, D]
            out[:, idx] = self.match(news[idx], e, "bkd,cd->bck")
        return out
