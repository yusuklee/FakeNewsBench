"""HDInt (KDD 2024) 원본 모듈 — HDInt_private/models/component/hdint_model.py 에서 이식.

BERT 는 여기 없다 (models/hdint.py 래퍼가 관리). 이 파일은 임베딩(raw content 1536 / keyword 2304)
을 입력받는 HDInt 본체만 포함한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DisentangleEncoder(nn.Module):
    """수식 2-4: DenseNet 스타일 3층 인코더."""

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc3 = nn.Linear(embed_dim * 2, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, c_j):
        z1 = self.act(self.drop(self.fc1(c_j)))
        z2 = self.act(self.drop(self.fc2(torch.cat([c_j, z1], dim=-1))))
        return self.act(self.drop(self.fc3(torch.cat([c_j, z2], dim=-1))))


class DisentangleDecoder(nn.Module):
    """수식 5-7: 3층 디코더."""

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc3 = nn.Linear(embed_dim * 2, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, h):
        d1 = self.act(self.drop(self.fc1(h)))
        d2 = self.act(self.drop(self.fc2(torch.cat([d1, h], dim=-1))))
        return self.act(self.drop(self.fc3(torch.cat([d2, h], dim=-1))))


def _dis_losses(classifier, recon_proj, x, part, free, lbl):
    """재구성 / 레이블 / 적대 손실 (수식 9-11). x: 입력, part: 분리 성분, free: 제거 표현."""
    out = {}
    recon = recon_proj(torch.cat([free, part], dim=-1))
    out["recon"] = 0.5 * F.mse_loss(recon, x)
    out["label"] = F.cross_entropy(classifier(part), lbl)
    y_hat = F.softmax(classifier(free), dim=-1)
    y_true = y_hat.gather(1, lbl.unsqueeze(1)).squeeze(1).clamp(0.05, 0.95)
    out["adversarial"] = (-1.0 / torch.log(y_true)).mean()
    return out


class PolarityDisentangler(nn.Module):
    def __init__(self, embed_dim: int, num_polarity_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = DisentangleEncoder(embed_dim, dropout)
        self.polarity_decoder = DisentangleDecoder(embed_dim, dropout)
        self.polarity_free_dec = DisentangleDecoder(embed_dim, dropout)
        self.recon_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_polarity_classes)

    def forward(self, e_j, polarity_labels=None, valid_mask=None):
        h = self.encoder(e_j)
        p_j = self.polarity_decoder(h)
        f_j = self.polarity_free_dec(h)
        loss_dict = {}
        if polarity_labels is not None:
            if valid_mask is not None:
                e_s, f_s, p_s, lbl = e_j[valid_mask], f_j[valid_mask], p_j[valid_mask], polarity_labels[valid_mask]
            else:
                e_s, f_s, p_s, lbl = e_j, f_j, p_j, polarity_labels
            if lbl.numel() > 0:
                loss_dict = _dis_losses(self.classifier, self.recon_proj, e_s, p_s, f_s, lbl)
        return f_j, p_j, loss_dict

    @staticmethod
    def total_loss(loss_dict):
        return sum(loss_dict.values()) if loss_dict else torch.tensor(0.0)


class VeracityDisentangler(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = DisentangleEncoder(embed_dim, dropout)
        self.veracity_decoder = DisentangleDecoder(embed_dim, dropout)
        self.clean_decoder = DisentangleDecoder(embed_dim, dropout)
        self.recon_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)  # 0 real / 1 fake

    def forward(self, f_j, veracity_labels=None, valid_mask=None):
        h = self.encoder(f_j)
        v_j = self.veracity_decoder(h)
        e_clean = self.clean_decoder(h)
        loss_dict = {}
        if veracity_labels is not None:
            if valid_mask is not None:
                f_s, c_s, v_s, lbl = f_j[valid_mask], e_clean[valid_mask], v_j[valid_mask], veracity_labels[valid_mask]
            else:
                f_s, c_s, v_s, lbl = f_j, e_clean, v_j, veracity_labels
            if lbl.numel() > 0:
                loss_dict = _dis_losses(self.classifier, self.recon_proj, f_s, v_s, c_s, lbl)
        return e_clean, v_j, loss_dict

    @staticmethod
    def total_loss(loss_dict):
        return sum(loss_dict.values()) if loss_dict else torch.tensor(0.0)

    def predict_veracity_from_f(self, f_j):
        h = self.encoder(f_j)
        v_j = self.veracity_decoder(h)
        return F.softmax(self.classifier(v_j), dim=-1)[:, 1]


class DisentanglingModule(nn.Module):
    def __init__(self, embed_dim: int, num_polarity_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.polarity_dis = PolarityDisentangler(embed_dim, num_polarity_classes, dropout)
        self.veracity_dis = VeracityDisentangler(embed_dim, dropout)

    def forward(self, e_j, polarity_labels=None, veracity_labels=None, valid_mask=None):
        f_j, _, pol = self.polarity_dis(e_j, polarity_labels, valid_mask)
        e_clean, _, ver = self.veracity_dis(f_j, veracity_labels, valid_mask)
        return e_clean, self.polarity_dis.total_loss(pol), self.veracity_dis.total_loss(ver)

    def predict_veracity(self, e_j):
        f_j, _, _ = self.polarity_dis(e_j)
        return self.veracity_dis.predict_veracity_from_f(f_j)


class InterestLearner(nn.Module):
    """수식 15-16: 1층 트랜스포머 (길이 유지)."""

    def __init__(self, embed_dim, num_heads, num_layers=1, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, padding_mask=None):
        return self.transformer(x, src_key_padding_mask=padding_mask)


class InterestAggregation(nn.Module):
    """수식 17-19: 후보를 쿼리로 하는 이중선형 어텐션 + FC 융합."""

    def __init__(self, embed_dim, user_embed_dim=32, dropout=0.1, use_user_id=False):
        super().__init__()
        self.W1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=False)
        fusion_in = embed_dim * 2 + (user_embed_dim if use_user_id else 0)
        self.fusion = nn.Linear(fusion_in, embed_dim)

    def forward(self, K_tilde, E_tilde, cand_keyword, cand_clean, mask=None, user_emb=None):
        qh = self.W1(cand_keyword).unsqueeze(1)
        ql = self.W2(cand_clean).unsqueeze(1)
        score_h = (K_tilde * qh).sum(-1)
        score_l = (E_tilde * ql).sum(-1)
        if mask is not None:
            score_h = score_h.masked_fill(mask, float("-inf"))
            score_l = score_l.masked_fill(mask, float("-inf"))
        alpha = torch.softmax(score_h, dim=1).unsqueeze(-1)
        beta = torch.softmax(score_l, dim=1).unsqueeze(-1)
        i_high = (alpha * K_tilde).sum(1)
        i_low = (beta * E_tilde).sum(1)
        parts = [i_high, i_low] + ([user_emb] if user_emb is not None else [])
        return self.fusion(torch.cat(parts, dim=-1))

    def forward_many(self, K_tilde, E_tilde, cand_keyword, cand_clean, mask=None, user_emb=None):
        """후보 C개를 한 번에: K/E_tilde [B,S,D], cand_* [C,D] -> score [B,C]  (full-ranking 용)."""
        qh = self.W1(cand_keyword)                                    # [C, D]
        ql = self.W2(cand_clean)
        score_h = torch.einsum("bsd,cd->bsc", K_tilde, qh)            # [B, S, C]
        score_l = torch.einsum("bsd,cd->bsc", E_tilde, ql)
        if mask is not None:
            m = mask.unsqueeze(-1)
            score_h = score_h.masked_fill(m, float("-inf"))
            score_l = score_l.masked_fill(m, float("-inf"))
        alpha = torch.softmax(score_h, dim=1)                         # [B, S, C]
        beta = torch.softmax(score_l, dim=1)
        i_high = torch.einsum("bsc,bsd->bcd", alpha, K_tilde)         # [B, C, D]
        i_low = torch.einsum("bsc,bsd->bcd", beta, E_tilde)
        parts = [i_high, i_low]
        if user_emb is not None:
            parts.append(user_emb.unsqueeze(1).expand(-1, i_high.size(1), -1))
        u = self.fusion(torch.cat(parts, dim=-1))                     # [B, C, D]
        return (u * cand_clean.unsqueeze(0)).sum(-1)                  # [B, C]


class HDIntCore(nn.Module):
    """HDInt 본체 (BERT 제외). 입력 = raw content(1536) / keyword(2304) 임베딩."""

    def __init__(self, content_in_dim=1536, keyword_in_dim=2304, embed_dim=256, num_heads=4,
                 num_polarity_classes=3, num_transformer_layers=1, dropout=0.1,
                 num_users=0, user_embed_dim=32):
        super().__init__()
        self.content_proj = nn.Linear(content_in_dim, embed_dim)          # 수식 1 (sigmoid)
        self.disentangler = DisentanglingModule(embed_dim, num_polarity_classes, dropout)
        self.keyword_proj = nn.Linear(keyword_in_dim, embed_dim)          # 수식 14
        self.high_learner = InterestLearner(embed_dim, num_heads, num_transformer_layers, dropout)
        self.low_learner = InterestLearner(embed_dim, num_heads, num_transformer_layers, dropout)
        self.use_user_id = num_users > 0
        if self.use_user_id:
            self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        self.aggregation = InterestAggregation(embed_dim, user_embed_dim, dropout, self.use_user_id)

    def proj_content(self, x):
        return torch.sigmoid(self.content_proj(x))

    @staticmethod
    def _score(u, cand_clean):
        return torch.sigmoid((u * cand_clean).sum(-1))                 # 수식 20

    def encode_context(self, ctx_content, ctx_keyword, padding_mask, ctx_polarity=None, ctx_veracity=None):
        """컨텍스트 분리 + 두 트랜스포머. 반환 K_tilde, E_tilde [B,S,D], pol_ctx, ver_ctx."""
        B, S = ctx_content.shape[:2]
        flat = self.proj_content(ctx_content.reshape(B * S, -1))
        pol = ctx_polarity.reshape(B * S) if ctx_polarity is not None else None
        ver = ctx_veracity.reshape(B * S) if ctx_veracity is not None else None
        valid = (~padding_mask).reshape(B * S) if padding_mask is not None else None
        clean, pol_ctx, ver_ctx = self.disentangler(flat, pol, ver, valid)
        ctx_clean = clean.reshape(B, S, -1)
        K_tilde = self.high_learner(self.keyword_proj(ctx_keyword), padding_mask)
        E_tilde = self.low_learner(ctx_clean, padding_mask)
        return K_tilde, E_tilde, pol_ctx, ver_ctx

    def forward(self, ctx_content, ctx_keyword, cand_content, cand_keyword, padding_mask=None,
                ctx_polarity=None, ctx_veracity=None, cand_polarity=None, cand_veracity=None, user_id=None):
        K_tilde, E_tilde, pol_ctx, ver_ctx = self.encode_context(ctx_content, ctx_keyword, padding_mask,
                                                                  ctx_polarity, ctx_veracity)
        cand_clean, pol_cand, ver_cand = self.disentangler(self.proj_content(cand_content), cand_polarity, cand_veracity)
        cand_kw = self.keyword_proj(cand_keyword)
        user_emb = self.user_embedding(user_id) if (self.use_user_id and user_id is not None) else None
        u = self.aggregation(K_tilde, E_tilde, cand_kw, cand_clean, padding_mask, user_emb)
        return self._score(u, cand_clean), pol_ctx, pol_cand, ver_ctx, ver_cand

    def score_batch(self, ctx_content, ctx_keyword, padding_mask, user_id, cand_cache, chunk=1024):
        """컨텍스트 한 번 인코딩 후 모든 후보 점수 [B, C]. cand_cache = (clean [C,D], keyword_proj [C,D])."""
        K_tilde, E_tilde, _, _ = self.encode_context(ctx_content, ctx_keyword, padding_mask)
        user_emb = self.user_embedding(user_id) if (self.use_user_id and user_id is not None) else None
        clean, kw = cand_cache
        outs = []
        for s in range(0, clean.size(0), chunk):
            outs.append(self.aggregation.forward_many(K_tilde, E_tilde, kw[s:s + chunk], clean[s:s + chunk],
                                                      padding_mask, user_emb))
        return torch.sigmoid(torch.cat(outs, dim=1))
