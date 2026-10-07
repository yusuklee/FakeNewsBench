"""NRMS (EMNLP'19). 1단계.
생략: 없음. 단어 임베딩은 words.pt (GloVe / Chinese Word Vectors) 로 초기화.
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
            a = a.masked_fill(~mask[..., None, None, :], -1e9)
        out = F.softmax(a, dim=-1) @ v                                                 # [.., H, T, d]
        return out.transpose(-2, -3).flatten(-2)


class AttentivePooling(nn.Module):
    """가산 어텐션 풀링. 전부 패딩이면 0."""

    def __init__(self, dim, query_dim=200):
        super().__init__()
        self.fc = nn.Linear(dim, query_dim)
        self.query = nn.Linear(query_dim, 1)

    def forward(self, x, mask=None):  # x [.., T, D] -> [.., D]
        a = self.query(torch.tanh(self.fc(x))).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        out = (F.softmax(a, dim=-1).unsqueeze(-1) * x).sum(-2)
        return out if mask is None else out * mask.any(-1, keepdim=True)


class NRMS(BaseModel):
    def __init__(self, data, cfg, device):
        super().__init__(data, cfg, device)
        words = self.load_file("words.pt")
        self.register_buffer("title", words["title"], persistent=False)   # [N+1, 30] 단어 번호
        self.word_emb = nn.Embedding.from_pretrained(words["emb"], freeze=False, padding_idx=0)
        heads, head_dim = cfg["num_heads"], cfg["head_dim"]
        dim = heads * head_dim
        self.drop = nn.Dropout(cfg["dropout"])
        self.word_att = SelfAttention(words["emb"].size(1), heads, head_dim)
        self.word_pool = AttentivePooling(dim, cfg["query_dim"])
        self.news_att = SelfAttention(dim, heads, head_dim)
        self.news_pool = AttentivePooling(dim, cfg["query_dim"])

    def news_vec(self, idx):  # [..] -> [.., D]
        w = self.title[idx]
        mask = w > 0
        x = self.drop(self.word_emb(w))
        x = self.drop(self.word_att(x, mask))
        return self.word_pool(x, mask)

    def user_vec(self, h, mask):  # h [B, L, D] 히스토리 뉴스 벡터 -> [B, D]
        return self.news_pool(self.drop(self.news_att(h, mask)), mask)

    # ---- 인터페이스
    def compute_loss(self, batch, stage=0):
        cand = torch.cat([batch["target"].unsqueeze(1), batch["neg"]], 1)   # [B, 1+K], 0번이 정답
        u = self.user_vec(self.news_vec(batch["ctx"]), batch["mask"])
        logits = (self.news_vec(cand) * u.unsqueeze(1)).sum(-1)
        loss = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))
        return loss, {}

    @torch.no_grad()
    def score(self, batch):
        news = self.all_news_vecs(self.news_vec)                             # [N+1, D]
        u = self.user_vec(news[batch["ctx"]], batch["mask"])
        return u @ news.T
