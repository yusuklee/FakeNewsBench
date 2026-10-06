"""RobustSentiRec (Sertkan et al., 2022). 1단계.

생략: 없음. 감성 점수는 sentiment.pt (원본 bert_sentiment 방식), 단어 임베딩은 words.pt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, dim):
        super().__init__()
        self.linear = nn.Linear(dim, query_dim)
        self.query = nn.Parameter(torch.empty(query_dim, 1).uniform_(-0.1, 0.1))

    def forward(self, x, pad):  # x [B, T, D], pad [B, T] True = 패딩 -> [B, D]
        a = (torch.tanh(self.linear(x)) @ self.query).squeeze(-1).masked_fill(pad, -1e9)
        return (F.softmax(a, dim=1).unsqueeze(-1) * x).sum(1)


class Encoder(nn.Module):
    """다중 헤드 셀프 어텐션 + 가산 어텐션 (원본 NewsEncoder / UserEncoder 공통 구조)."""

    def __init__(self, dim, heads, query_dim):
        super().__init__()
        self.att = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.pool = AdditiveAttention(query_dim, dim)

    def forward(self, x, pad, dropout=0.0):
        pad = pad.clone()
        pad[:, 0] &= ~pad.all(1)        # 전부 패딩인 행은 첫 칸을 열어 NaN 방지
        x, _ = self.att(x, x, x, key_padding_mask=pad)
        return self.pool(F.dropout(x, dropout, self.training), pad)


class RobustSentiRec(BaseModel):
    def __init__(self, data, cfg, device):
        super().__init__(data, cfg, device)
        words = self.load_file("words.pt")
        self.register_buffer("title", words["title"], persistent=False)            # [N+1, 30]
        self.register_buffer("senti", self.load_file("sentiment.pt"), persistent=False)  # [N+1]
        self.word_emb = nn.Embedding.from_pretrained(words["emb"], freeze=False, padding_idx=0)
        dim = words["emb"].size(1)
        self.p = cfg["dropout"]
        self.news_encoder = Encoder(dim, cfg["num_heads"], cfg["query_dim"])
        self.user_encoder = Encoder(dim, cfg["num_heads"], cfg["query_dim"])
        self.sentiment_aware = nn.Linear(dim + 1, dim)

    def news_vec(self, idx):  # [..] -> [.., D]  제목 벡터에 감성 점수를 붙여 선형 변환
        w = self.title[idx.flatten()]
        x = F.dropout(self.word_emb(w), self.p, self.training)
        v = self.news_encoder(x, w == 0, self.p)
        v = self.sentiment_aware(torch.cat([v, self.senti[idx.flatten()].unsqueeze(-1)], -1))
        return v.view(*idx.shape, -1)

    # ---- 인터페이스
    def compute_loss(self, batch, stage=0):
        cand = torch.cat([batch["target"].unsqueeze(1), batch["neg"]], 1)          # [B, 1+K], 0번이 정답
        u = self.user_encoder(self.news_vec(batch["ctx"]), ~batch["mask"])
        y = torch.sigmoid((self.news_vec(cand) * u.unsqueeze(1)).sum(-1))            # 원본: sigmoid 뒤에 CE
        loss_rec = F.cross_entropy(y, torch.zeros(len(y), dtype=torch.long, device=y.device))
        # 감성 다양성 손실: 유저 히스토리 평균 감성과 같은 방향의 후보에 높은 점수를 주면 벌점
        m = batch["mask"].float()
        s_mean = (self.senti[batch["ctx"]] * m).sum(1) / m.sum(1).clamp(min=1)
        loss_senti = F.relu(s_mean.unsqueeze(1) * self.senti[cand] * y).mean()
        loss = loss_rec + self.cfg["mu"] * loss_senti
        return loss, {"L_rec": loss_rec.item(), "L_senti": loss_senti.item()}

    @torch.no_grad()
    def score(self, batch):
        news = self.all_news_vecs(self.news_vec)                                    # [N+1, D]
        u = self.user_encoder(news[batch["ctx"]], ~batch["mask"])
        return u @ news.T
