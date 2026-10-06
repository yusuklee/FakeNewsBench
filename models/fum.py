"""FUM (SIGIR'22). 1단계.

생략: 없음. 본문(content)은 description 단어, 엔티티 임베딩은 원본처럼 처음부터 학습.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel

NEG = -1e9


class SelfAttention(nn.Module):
    """다중 헤드 셀프 어텐션 (원본 Keras Attention: Q/K/V 투영만, 출력 투영 없음)."""

    def __init__(self, in_dim, heads, head_dim):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.q = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.k = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.v = nn.Linear(in_dim, heads * head_dim, bias=False)

    def forward(self, x, mask):  # x [.., T, D], mask [.., T] True = 실제 -> [.., T, heads*head_dim]
        def split(t):
            return t.view(*t.shape[:-1], self.heads, self.head_dim).transpose(-2, -3)  # [.., H, T, d]

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        a = (q @ k.transpose(-1, -2) / self.head_dim ** 0.5).masked_fill(~mask[..., None, None, :], NEG)
        return (F.softmax(a, dim=-1) @ v).transpose(-2, -3).flatten(-2)


class AttentivePooling(nn.Module):
    """Dropout + 가산 어텐션 풀링. 전부 패딩이면 0."""

    def __init__(self, dim, dropout, query_dim=200):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, query_dim)
        self.query = nn.Linear(query_dim, 1)

    def forward(self, x, mask):  # x [.., T, D] -> [.., D]
        x = self.drop(x)
        a = self.query(torch.tanh(self.fc(x))).squeeze(-1).masked_fill(~mask, NEG)
        return (F.softmax(a, dim=-1).unsqueeze(-1) * x).sum(-2) * mask.any(-1, keepdim=True)


class Fastformer(nn.Module):
    """Fastformer: 전역 query -> 전역 key -> query 와 원소곱. O(T)."""

    def __init__(self, in_dim, heads, head_dim):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        dim = heads * head_dim
        self.q = nn.Linear(in_dim, dim, bias=False)
        self.k = nn.Linear(in_dim, dim, bias=False)
        self.wa = nn.Linear(dim, heads, bias=False)
        self.wb = nn.Linear(dim, heads, bias=False)
        self.wp = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask):  # x [B, T, D], mask [B, T] -> [B, T, heads*head_dim]
        B, T, _ = x.shape
        q, k = self.q(x), self.k(x)                                                    # [B, T, H*d]
        qh, kh = q.view(B, T, self.heads, -1), k.view(B, T, self.heads, -1)            # [B, T, H, d]
        pad = ~mask.unsqueeze(-1)                                                      # [B, T, 1]
        alpha = F.softmax((self.wa(q) / self.head_dim ** 0.5).masked_fill(pad, NEG), dim=1)   # [B, T, H]
        gq = (alpha.unsqueeze(-1) * qh).sum(1, keepdim=True)                           # [B, 1, H, d] 전역 query
        p = kh * gq
        beta = F.softmax((self.wb(p.flatten(2)) / self.head_dim ** 0.5).masked_fill(pad, NEG), dim=1)
        gk = (beta.unsqueeze(-1) * p).sum(1, keepdim=True)                             # [B, 1, H, d] 전역 key
        return self.wp((gk * qh).flatten(2)) + q


class FUM(BaseModel):
    def __init__(self, data, cfg, device):
        super().__init__(data, cfg, device)
        words = self.load_file("words.pt")
        self.register_buffer("title", words["title"], persistent=False)                          # [N+1, 30]
        self.register_buffer("content", words["description"], persistent=False)                  # [N+1, 50]
        self.register_buffer("entity", self.load_file("entity.pt"), persistent=False)            # [N+1, 5]
        self.register_buffer("category", self.load_file("category.pt"), persistent=False)        # [N+1]
        self.register_buffer("subcategory", self.load_file("subcategory.pt"), persistent=False)  # [N+1]
        meta, p, emb = data["meta"], cfg["dropout"], words["emb"]
        n_cat, n_sub, wd = meta["num_category"] + 1, meta["num_subcategory"] + 1, emb.size(1)
        self.drop = nn.Dropout(p)

        # 뉴스 인코더: 제목 + 본문 + 카테고리 + 서브카테고리 + 엔티티
        self.title_emb = nn.Embedding.from_pretrained(emb.clone(), freeze=False, padding_idx=0)
        self.title_att, self.title_pool = SelfAttention(wd, 20, 20), AttentivePooling(400, p)
        self.content_emb = nn.Embedding.from_pretrained(emb.clone(), freeze=False, padding_idx=0)
        self.content_att, self.content_pool = SelfAttention(wd, 20, 20), AttentivePooling(400, p)
        self.entity_emb = nn.Embedding(meta["num_entity"] + 1, 300, padding_idx=0)
        self.entity_att, self.entity_pool = SelfAttention(300, 5, 40), AttentivePooling(200, p)
        self.cat_emb, self.cat_fc = nn.Embedding(n_cat, 128, padding_idx=0), nn.Linear(128, 128)
        self.sub_emb, self.sub_fc = nn.Embedding(n_sub, 128, padding_idx=0), nn.Linear(128, 128)
        self.news_fc = nn.Linear(400 + 400 + 128 + 128 + 200, 400)

        # 굵은 유저 모델 (뉴스 벡터 단위)
        self.user_att, self.user_pool, self.user_fc = SelfAttention(400, 20, 20), AttentivePooling(400, p), nn.Linear(400, 370)

        # 세밀한 유저 모델 (히스토리 전체의 단어를 한 줄로 이어 Fastformer)
        self.f_word_emb = nn.Embedding.from_pretrained(emb.clone(), freeze=False, padding_idx=0)
        self.f_cat_emb = nn.Embedding(n_cat, wd, padding_idx=0)
        self.f_sub_emb = nn.Embedding(n_sub, wd, padding_idx=0)
        self.fastformer = Fastformer(wd, 3, 40)
        self.f_word_pool, self.f_news_fc = AttentivePooling(120, p), nn.Linear(360, 40)
        self.f_user_pool, self.f_user_fc = AttentivePooling(40, p), nn.Linear(40, 30)

    def _text(self, w, emb, att, pool):
        return pool(self.drop(att(self.drop(emb(w)), w > 0)), w > 0)

    def news_vec(self, idx):  # [..] -> [.., 400]
        t = self._text(self.title[idx], self.title_emb, self.title_att, self.title_pool)
        c = self._text(self.content[idx], self.content_emb, self.content_att, self.content_pool)
        e = self._text(self.entity[idx], self.entity_emb, self.entity_att, self.entity_pool)
        v = self.drop(self.cat_fc(self.cat_emb(self.category[idx])))
        s = self.drop(self.sub_fc(self.sub_emb(self.subcategory[idx])))
        return F.relu(self.news_fc(torch.cat([t, c, v, s, e], -1)))

    def user_vec(self, ctx, mask, hist):  # ctx [B, L], hist [B, L, 400] 히스토리 뉴스 벡터 -> [B, 400]
        coarse = self.user_fc(self.user_pool(self.user_att(self.drop(hist), mask), mask))       # [B, 370]

        B, L = ctx.shape
        w = self.title[ctx]                                                                      # [B, L, T]
        x = torch.cat([self.f_word_emb(w), self.f_cat_emb(self.category[ctx]).unsqueeze(2),
                       self.f_sub_emb(self.subcategory[ctx]).unsqueeze(2)], 2)                   # [B, L, T+2, 300]
        m = torch.cat([w > 0, mask.unsqueeze(-1).expand(-1, -1, 2)], 2)                          # [B, L, T+2]
        x = self.drop(self.fastformer(self.drop(x.flatten(1, 2)), m.flatten(1)))                 # [B, L*(T+2), 120]
        x = x.view(B, L, -1, x.size(-1))
        n = torch.cat([self.f_word_pool(x[:, :, :-2], w > 0), x[:, :, -2], x[:, :, -1]], -1)     # [B, L, 360]
        n = self.drop(self.f_news_fc(n))
        fine = self.drop(self.f_user_fc(self.f_user_pool(n, mask)))                              # [B, 30]
        return torch.cat([coarse, fine], -1)

    # ---- 인터페이스
    def compute_loss(self, batch, stage=0):
        cand = torch.cat([batch["target"].unsqueeze(1), batch["neg"]], 1)          # [B, 1+K], 0번이 정답
        u = self.user_vec(batch["ctx"], batch["mask"], self.news_vec(batch["ctx"]))
        logits = (self.drop(self.news_vec(cand)) * u.unsqueeze(1)).sum(-1)
        loss = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))
        return loss, {}

    @torch.no_grad()
    def score(self, batch):
        news = self.all_news_vecs(self.news_vec)                                    # [N+1, 400]
        return self.user_vec(batch["ctx"], batch["mask"], news[batch["ctx"]]) @ news.T
