"""CAUM (SIGIR'22). 1단계.

생략: MIND 사전학습 엔티티 벡터 (엔티티 임베딩을 처음부터 학습).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


class SelfAttention(nn.Module):
    """다중 헤드 셀프 어텐션 (원본 Keras Attention: Q/K/V 투영만, 출력 투영 없음)."""

    def __init__(self, in_dim, heads, head_dim):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.q = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.k = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.v = nn.Linear(in_dim, heads * head_dim, bias=False)

    def forward(self, x, mask=None):  # x [.., T, D], mask [.., T] True = 실제 -> [.., T, heads*head_dim]
        def split(t):
            return t.view(*t.shape[:-1], self.heads, self.head_dim).transpose(-2, -3)  # [.., H, T, d]

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        a = q @ k.transpose(-1, -2) / self.head_dim ** 0.5                             # [.., H, T, T]
        if mask is not None:
            a = a.masked_fill(~mask[..., None, None, :], torch.finfo(a.dtype).min)
        out = F.softmax(a, dim=-1) @ v                                                 # [.., H, T, d]
        return out.transpose(-2, -3).flatten(-2)


class AttentivePooling(nn.Module):
    """Dropout + 가산 어텐션 풀링. 전부 패딩이면 0."""

    def __init__(self, dim, dropout, query_dim=200):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, query_dim)
        self.query = nn.Linear(query_dim, 1)

    def forward(self, x, mask):  # x [.., T, D] -> [.., D]
        x = self.drop(x)
        a = self.query(torch.tanh(self.fc(x))).squeeze(-1)
        a = a.masked_fill(~mask, torch.finfo(a.dtype).min)
        return (F.softmax(a, dim=-1).unsqueeze(-1) * x).sum(-2) * mask.any(-1, keepdim=True)


class CAUM(BaseModel):
    def __init__(self, data, cfg, device):
        super().__init__(data, cfg, device)
        words = self.load_file("words.pt")
        self.register_buffer("title", words["title"], persistent=False)                    # [N+1, 30]
        self.register_buffer("entity", self.load_file("entity.pt"), persistent=False)      # [N+1, 5]
        self.register_buffer("category", self.load_file("category.pt"), persistent=False)  # [N+1]
        meta, p, dim = data["meta"], cfg["dropout"], cfg["news_dim"]
        self.drop = nn.Dropout(p)

        # 뉴스 인코더: 제목 + 엔티티 + 카테고리
        self.word_emb = nn.Embedding.from_pretrained(words["emb"], freeze=False, padding_idx=0)
        self.word_att = SelfAttention(words["emb"].size(1), 20, 20)
        self.word_pool = AttentivePooling(400, p)
        self.entity_emb = nn.Embedding(meta["num_entity"] + 1, cfg["entity_dim"], padding_idx=0)
        self.entity_att = SelfAttention(cfg["entity_dim"], 4, 40)
        self.entity_pool = AttentivePooling(160, p)
        self.cat_emb = nn.Embedding(meta["num_category"] + 1, 100, padding_idx=0)
        self.cat_emb.weight.requires_grad_(False)      # 원본: trainable=False
        self.cat_fc = nn.Linear(100, 100)
        self.news_fc = nn.Linear(400 + 160 + 100, dim)

        # 후보 인지 유저 모델링
        self.cnn_fc = nn.Linear(dim * 4, dim)          # [왼쪽, 자기, 오른쪽, 후보]
        self.self_fc = nn.Linear(dim * 2, dim)         # [후보, 자기]
        self.self_att = SelfAttention(dim, 20, dim // 20)
        self.out_fc = nn.Linear(dim * 2, dim)
        self.dense_att = nn.Sequential(nn.Linear(dim * 2, 400), nn.Tanh(), nn.Linear(400, 256), nn.Tanh(),
                                       nn.Linear(256, 1))

    def news_vec(self, idx):  # [..] -> [.., D]
        w, e = self.title[idx], self.entity[idx]
        t = self.drop(self.word_att(self.drop(self.word_emb(w)), w > 0))
        t = self.word_pool(t, w > 0)
        en = self.drop(self.entity_att(self.drop(self.entity_emb(e)), e > 0))
        en = self.entity_pool(en, e > 0)
        c = self.cat_fc(self.drop(self.cat_emb(self.category[idx])))
        return self.news_fc(torch.cat([t, en, c], -1))

    def match(self, cand, hist, mask):
        """cand [B, C, D] 후보, hist [B, L, D] 히스토리, mask [B, L] -> 점수 [B, C]"""
        C, L = cand.size(1), hist.size(1)
        uv = self.drop(hist).unsqueeze(1).expand(-1, C, -1, -1)                    # [B, C, L, D]
        can = cand.unsqueeze(2).expand(-1, -1, L, -1)                              # [B, C, L, D]
        m = mask.unsqueeze(1).expand(-1, C, -1)                                    # [B, C, L]
        cnn = self.cnn_fc(torch.cat([uv.roll(1, 2), uv, uv.roll(-1, 2), can], -1))
        att = self.self_att(self.self_fc(torch.cat([can, uv], -1)), m)
        h = self.out_fc(self.drop(torch.cat([cnn, att], -1)))                      # [B, C, L, D]
        a = self.dense_att(torch.cat([h, can], -1)).squeeze(-1)
        a = a.masked_fill(~m, torch.finfo(a.dtype).min)
        user = (F.softmax(a, dim=-1).unsqueeze(-1) * h).sum(2)                     # [B, C, D] 후보별 유저 벡터
        return (user * self.drop(cand)).sum(-1)

    # ---- 인터페이스
    def compute_loss(self, batch, stage=0):
        cand = torch.cat([batch["target"].unsqueeze(1), batch["neg"]], 1)          # [B, 1+K], 0번이 정답
        logits = self.match(self.news_vec(cand), self.news_vec(batch["ctx"]), batch["mask"])
        loss = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))
        return loss, {}

    @torch.no_grad()
    def score(self, batch):
        news = self.all_news_vecs(self.news_vec)                                    # [N+1, D]
        hist, mask = news[batch["ctx"]], batch["mask"]
        chunk = max(1, self.cfg["score_pairs"] // len(hist))                        # 후보별 유저 벡터라 후보를 나눠 계산
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            out = [self.match(c.unsqueeze(0).expand(len(hist), -1, -1), hist, mask) for c in news.split(chunk)]
        return torch.cat(out, 1).float()
