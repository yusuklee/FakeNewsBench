"""Rec4Mit (WWW'22). 1단계.
평가 시 자체 분류기로 fake 후보 제외 (filter_fake).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


# ───────────────────────── Layer 1 · 임베딩 ─────────────────────────
class EmbeddingLayer(nn.Module):
    def __init__(self, emb, id_dim=128, out_dim=256):
        super().__init__()
        num_news, meta_dim = emb.shape
        self.id_emb = nn.Embedding(num_news, id_dim, padding_idx=0)
        self.meta_emb = nn.Embedding(num_news, meta_dim, padding_idx=0)
        with torch.no_grad():
            self.meta_emb.weight.copy_(emb)
        self.fc_v = nn.Linear(id_dim + meta_dim, out_dim)

    def forward(self, idx):
        return self.fc_v(torch.cat([self.id_emb(idx), self.meta_emb(idx)], -1))


# ───────────────────────── Layer 2 · 분리 ─────────────────────────
class _Dense3(nn.Module):
    """3단 Dense + skip-connection, LeakyReLU(0.1)."""

    def __init__(self, in_dim, z1_dim, z2_dim, h_dim):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, z1_dim)
        self.dense2 = nn.Linear(in_dim + z1_dim, z2_dim)
        self.dense3 = nn.Linear(in_dim + z2_dim, h_dim)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, in_):
        z1 = self.act(self.dense1(in_))
        z2 = self.act(self.dense2(torch.cat([in_, z1], -1)))
        return self.act(self.dense3(torch.cat([in_, z2], -1)))


class Encoder(_Dense3):
    def __init__(self, in_dim=256, z1_dim=256, z2_dim=256, h_dim=256):
        super().__init__(in_dim, z1_dim, z2_dim, h_dim)


class EventDecoder(_Dense3):
    def __init__(self, in_dim=256, z1_dim=256, z2_dim=256, h_dim=128):
        super().__init__(in_dim, z1_dim, z2_dim, h_dim)


class VeracityDecoder(_Dense3):
    def __init__(self, in_dim=256, z1_dim=256, z2_dim=256, h_dim=128):
        super().__init__(in_dim, z1_dim, z2_dim, h_dim)
        self.denseL = nn.Linear(h_dim, 1)

    def forward(self, in_):
        l_i = super().forward(in_)
        return l_i, self.denseL(l_i).squeeze(-1)  # sigmoid(logit) = y~ (Eq 8)


class DisentangleLoss(nn.Module):
    def __init__(self, e_dim=128):
        super().__init__()
        self.dense_e = nn.Linear(e_dim, 1)  # Eq 11 적대 예측기

    def forward(self, v, e, l, v_sig_input, y, mask=None):
        loss_r = 0.5 * ((torch.cat([e, l], -1) - v) ** 2).mean(-1)                      # Eq 9
        loss_l = F.binary_cross_entropy_with_logits(v_sig_input, y, reduction="none")  # Eq 10
        adv_err = F.binary_cross_entropy_with_logits(self.dense_e(e).squeeze(-1), y, reduction="none")
        loss_a = 1.0 / adv_err.clamp(min=1e-6)                                         # Eq 11
        losses = loss_l + loss_r + loss_a                                              # Eq 12
        if mask is not None:
            total = (losses * mask).sum() / mask.sum().clamp(min=1)
        else:
            total = losses.mean()
        return total, {"L_r": loss_r.mean().item(), "L_l": loss_l.mean().item(), "L_a": loss_a.mean().item()}


# ───────────────────────── Layer 3 · 사건 전이 / 예측 ─────────────────────────
class EventDetector(nn.Module):
    def __init__(self, e_dim=128, events=20):
        super().__init__()
        self.W1 = nn.Linear(e_dim, events, bias=False)

    def forward(self, e):  # [B,L,e] -> beta [B,L,K], e_split [B,L,K,e]
        beta = F.softmax(self.W1(e), dim=-1)
        return beta, beta.unsqueeze(-1) * e.unsqueeze(-2)


class EventTransitionNet(nn.Module):
    def __init__(self, e_dim=128, events=20, ctx_len=4, pos_dim=32, attn_dim=64, user_dim=128):
        super().__init__()
        self.W2 = nn.Linear(e_dim + pos_dim, attn_dim)
        self.W3 = nn.Linear(attn_dim, 1, bias=False)
        self.W4 = nn.Linear(e_dim + user_dim, e_dim)
        self.pos_emb = nn.Embedding(ctx_len, pos_dim)

    def build_R(self, e, e_split, pad_mask=None):
        b = e.size(0)
        p = self.pos_emb.weight
        f = torch.cat([e, p.unsqueeze(0).expand(b, -1, -1)], -1)
        gamma = self.W3(torch.tanh(self.W2(f))).squeeze(-1)
        if pad_mask is not None:
            gamma = gamma.masked_fill(~pad_mask, -1e9)
        gamma = F.softmax(gamma, dim=-1)
        R = torch.einsum("bl,blkd->bkd", gamma, e_split)  # Eq 15~16
        return R, gamma

    def activate(self, R, e_t, u):
        delta = F.softmax(torch.bmm(e_t, R.transpose(1, 2)), -1)  # Eq 17
        c = torch.einsum("bck,bkd->bcd", delta, R)                 # Eq 18
        u = u.unsqueeze(1).expand(-1, c.size(1), -1)
        c_u = torch.tanh(self.W4(torch.cat([c, u], -1)))           # Eq 19
        return c_u, delta


class NextNewsPredictor(nn.Module):
    def forward(self, c_u, e_t):
        return (c_u * e_t).sum(-1)  # Eq 20


# ───────────────────────── 전체 모델 ─────────────────────────
class Rec4Mit(BaseModel):
    def __init__(self, data, cfg, device):
        super().__init__(data, cfg, device)
        emb = self.full_emb()  # [N+1, 1536]
        if cfg.get("normalize_emb", True):  # 원본 sentence-transformers normalize_embeddings=True 재현 (필드별 단위 벡터)
            t, d = emb[:, :768], emb[:, 768:]
            emb = torch.cat([F.normalize(t, dim=-1), F.normalize(d, dim=-1)], dim=-1)
        k = cfg.get("k_events", 20)
        v_dim = cfg.get("v_dim", 256)
        e_dim = cfg.get("e_dim", 128)
        user_dim = cfg.get("user_dim", 128)
        self.ctx_len = cfg["max_len"]

        self.embedding = EmbeddingLayer(emb, out_dim=v_dim)
        self.encoder = Encoder(in_dim=v_dim, h_dim=v_dim)
        self.event_dec = EventDecoder(in_dim=v_dim, h_dim=e_dim)
        self.veracity_dec = VeracityDecoder(in_dim=v_dim, h_dim=e_dim)
        self.detector = EventDetector(e_dim, k)
        self.transition = EventTransitionNet(e_dim, k, self.ctx_len, user_dim=user_dim)
        self.predictor = NextNewsPredictor()
        self.user_emb = nn.Embedding(self.num_users + 1, user_dim, padding_idx=self.num_users)
        self.dis_loss = DisentangleLoss(e_dim)

    # ---- 원본 forward 구성요소
    def disentangle(self, ids):
        v = self.embedding(ids)
        h = self.encoder(v)
        e = self.event_dec(h)
        l, logit = self.veracity_dec(h)
        return v, e, l, logit

    def _user_vec(self, seq_ids, pad_mask):
        """컨텍스트 -> (e_s, R) ; 후보와 결합은 호출부에서."""
        v_s, e_s, l_s, lg_s = self.disentangle(seq_ids)
        _, e_split = self.detector(e_s)
        R, _ = self.transition.build_R(e_s, e_split, pad_mask)
        return (v_s, e_s, l_s, lg_s), R

    # ---- 인터페이스
    def compute_loss(self, batch, stage=0):
        seq, pad_mask = self.left_pad(batch["ctx"], batch["mask"])
        cand = torch.cat([batch["target"].unsqueeze(1), batch["neg"]], 1)  # [B, 1+K]
        u = batch["user"]

        ctx_out, R = self._user_vec(seq, pad_mask)
        v_c, e_c, l_c, lg_c = self.disentangle(cand)
        c_u, _ = self.transition.activate(R, e_c, self.user_emb(u))
        logits = self.predictor(c_u, e_c)  # [B, 1+K]

        target = torch.zeros_like(logits)
        target[:, 0] = 1
        loss_p = F.binary_cross_entropy_with_logits(logits, target)

        y_s = batch["ctx_label"]
        y_s, _ = self.left_pad(y_s.long(), batch["mask"])  # 라벨도 같은 순서로 이동
        y_s = y_s.float()
        y_c = batch["neg_label"]
        y_c = torch.cat([batch["target_label"].unsqueeze(1), y_c], 1)

        loss_d = self.dis_loss(*ctx_out, y_s, pad_mask.float())[0] + self.dis_loss(v_c, e_c, l_c, lg_c, y_c)[0]
        loss = loss_p + loss_d
        return loss, {"L_p": loss_p.item(), "L_d": loss_d.item()}

    @torch.no_grad()
    def score(self, batch):
        seq, pad_mask = self.left_pad(batch["ctx"], batch["mask"])
        u = batch["user"]
        B = seq.size(0)
        cand = torch.arange(0, self.num_news + 1, device=seq.device)
        _, e_c, _, _ = self.disentangle(cand)                  # [N+1, e]
        _, R = self._user_vec(seq, pad_mask)
        ec = e_c.unsqueeze(0).expand(B, -1, -1)
        c_u, _ = self.transition.activate(R, ec, self.user_emb(u))
        return self.predictor(c_u, ec)                        # [B, N+1]

    @torch.no_grad()
    def candidate_mask(self):
        if not self.cfg.get("filter_fake", True):
            return self.pc
        was_training = self.training
        self.eval()
        cand = torch.arange(0, self.num_news + 1, device=self.device)
        _, _, _, lg_c = self.disentangle(cand)
        mask = torch.sigmoid(lg_c) >= self.cfg.get("fake_threshold", 0.5)
        mask[0] = True
        if was_training:
            self.train()
        return mask | self.pc
