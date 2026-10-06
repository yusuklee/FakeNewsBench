"""PRISM (SIGIR'25). 2단계: IB 분류기 -> 조건부 디퓨전 (CFG).

생략: P_c/P_u 4:6 분리 (분류기가 전체 뉴스로 학습).
"""

import math
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

from models.base import BaseModel


# ---- 원본 구성요소 (PRISM_private/models/component/guided_diffusion.py 이식)

# ----------------------------------------------------------------------------- diffusion

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


class Diffusion(nn.Module):
    """조건부 디퓨전 추천기 (Phase 2)."""

    def __init__(self, news_emb: torch.Tensor, input_dim=1536, hidden_size=128, timesteps=200,
                 beta_start=0.1, beta_end=0.1, hyper_w=0.1,
                 max_len=5, p=0.1, dropout=0.1, num_heads=4, tau=0.07):
        super().__init__()
        self.timesteps = timesteps
        self.w = hyper_w
        self.p = p
        self.tau = tau
        self.register_buffer("news_emb", news_emb)            # [N+1, input_dim], row 0 = 패딩
        self.model = ConditionNet(input_dim=input_dim, hidden_size=hidden_size, state_size=max_len,
                                  dropout=dropout, num_heads=num_heads, max_len=max_len)

        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        reg = lambda n, v: self.register_buffer(n, v.float())
        reg("betas", betas)
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        reg("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        reg("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        reg("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    # ---- 유틸
    def lookup(self, ids):
        return self.news_emb[ids]

    def encoded_library(self):
        """전체 뉴스 테이블을 hidden 차원으로 투영 [N+1, hidden]."""
        return self.model.content_reflect(self.news_emb)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # ---- 학습
    def p_losses(self, seq, mask, target, authenticity, noise=None):
        """seq [B,L] 뉴스 idx (오른쪽 0 패딩), mask [B,L] bool, target [B], authenticity [1,input_dim] (e_real)."""
        len_seq = mask.sum(1)                                              # [B]
        content_features, cmask = self.model.content_encoder(self.lookup(seq), mask)
        x_start = self.model.content_reflect(self.lookup(target))          # [B, hidden]
        h, state_hidden = self.model.cacu_h(seq, self.lookup(seq), len_seq, self.p, mask)

        t = torch.randint(0, self.timesteps, (seq.size(0),), device=seq.device).long()
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        auth = self.model.content_reflect(authenticity)                    # [1, hidden]
        predicted_noise = self.model(x_noisy, h, t, content_features, state_hidden, auth, cmask)

        sa = extract(self.sqrt_alphas_cumprod, t, x_noisy.shape)
        s1 = extract(self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
        predicted_x0 = (x_noisy - s1 * predicted_noise) / sa

        encoded = self.encoded_library()                                   # [N+1, hidden]
        scores = F.normalize(predicted_x0, dim=-1) @ F.normalize(encoded, dim=-1).T / self.tau
        loss = F.mse_loss(noise, predicted_noise)
        return loss, scores

    # ---- 추론
    @torch.no_grad()
    def p_sample(self, x, h, t, t_index, content_features, state_hidden, auth0, auth1, cmask, gen=None):
        predicted_noise = ((1 + self.w) * self.model(x, h, t, content_features, state_hidden, auth0, cmask)
                           - self.w * self.model.forward_uncon(x, t, auth1))
        sa = extract(self.sqrt_alphas_cumprod, t, x.shape)
        s1 = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        x_start = (x - s1 * predicted_noise) / sa
        model_mean = (extract(self.posterior_mean_coef1, t, x.shape) * x_start
                      + extract(self.posterior_mean_coef2, t, x.shape) * x)
        if t_index == 0:
            return model_mean
        var = extract(self.posterior_variance, t, x.shape)
        return model_mean + torch.sqrt(var) * torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def sample(self, seq, mask, authenticity_0, authenticity_1):
        """-> (x0 [B,hidden], scores [B,N+1] cosine)"""
        len_seq = mask.sum(1)
        content_features, cmask = self.model.content_encoder(self.lookup(seq), mask)
        h, state_hidden = self.model.predict(seq, self.lookup(seq), len_seq, mask)
        # 평가 재현성: 노이즈는 고정 시드 generator 에서 (학습 RNG 와 분리)
        gen = torch.Generator(device=h.device).manual_seed(int(getattr(self, "eval_seed", 42)))
        x = torch.randn(h.shape, generator=gen, device=h.device, dtype=h.dtype)
        auth0 = self.model.content_reflect(authenticity_0)
        B = h.shape[0]
        for n in reversed(range(self.timesteps)):
            t = torch.full((B,), n, device=seq.device, dtype=torch.long)
            x = self.p_sample(x, h, t, n, content_features, state_hidden, auth0, authenticity_1, cmask, gen)
        encoded = self.encoded_library()
        scores = F.normalize(x, dim=-1) @ F.normalize(encoded, dim=-1).T
        return x, scores


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=time.device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ConditionNet(nn.Module):
    """조건부 denoiser (seqattn). 768 → input_dim 일반화."""

    def __init__(self, input_dim=1536, hidden_size=128, state_size=5, dropout=0.1,
                 num_heads=4, max_len=5):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.content_reflect = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.LeakyReLU(), norm_layer(hidden_size),
            nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size))
        self.content_pos = class_token_pos_embed(hidden_size, max_len)
        self.content_embed = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)

        self.none_embedding = nn.Embedding(1, hidden_size)
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(state_size, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)

        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.mh_attn_1 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.cross_attn_1 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.cross_attn_2 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)

        self.nn_1 = nn.Linear(hidden_size, hidden_size)
        self.nn_2 = nn.Linear(2 * hidden_size, hidden_size)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size), nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(), nn.Linear(hidden_size * 2, hidden_size))
        self.init_weights()

    def init_weights(self):
        for layer in self.content_reflect:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    # ---- 조건부 denoising
    def forward(self, x, h, step, content=None, state_hidden=None, authenticity=None, mask=None):
        content = self.nn_1(content)                                   # C_k [B, L+1, H]
        authenticity = self.nn_1(authenticity)                         # e_real [1, H]
        cos = F.cosine_similarity(content, authenticity.unsqueeze(1), dim=-1)
        gate = (1 + torch.tanh(cos)) / 2                               # A_k
        content = content * gate.unsqueeze(-1) + authenticity.unsqueeze(1) * (1 - gate).unsqueeze(-1)

        t = self.step_mlp(step)
        x = x.squeeze(1) if x.dim() == 3 else x
        x_t = self.nn_2(torch.cat((x, t), dim=1)).unsqueeze(1)
        e_hat = self.cross_attn_1(x_t, content, content, mask)
        c_hat = self.cross_attn_2(e_hat, state_hidden, state_hidden, mask[:, 1:])
        h_hat = self.mh_attn_1(c_hat, h.unsqueeze(1), h.unsqueeze(1))
        return self.feed_forward(h_hat).squeeze(1)

    # ---- 무조건 denoising (h 대신 phi, e_fake 조건)
    def forward_uncon(self, x, step, authenticity):
        B = x.shape[0]
        phi = self.none_embedding.weight.view(1, self.hidden_size).expand(B, -1)
        t = self.step_mlp(step)
        auth = self.nn_1(self.content_reflect(authenticity)).expand(B, -1)
        x_t = self.nn_2(torch.cat((x, t), dim=1)).unsqueeze(1)
        auth_kv = auth.unsqueeze(1)
        phi_kv = phi.unsqueeze(1)
        e_hat = self.cross_attn_1(x_t, auth_kv, auth_kv)
        c_hat = self.cross_attn_2(e_hat, phi_kv, phi_kv)
        c_hat = self.mh_attn_1(c_hat, phi_kv, phi_kv)
        return self.feed_forward(c_hat).squeeze(1)

    def content_encoder(self, text, mask):
        """text [B,L,input_dim], mask [B,L] bool -> (features [B,L+1,H], mask [B,L+1])"""
        text = self.content_reflect(text)
        cls_mask = torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=-1)
        text = self.content_pos(text)
        return self.content_embed(text, text, text, mask), mask

    def _encode_seq(self, states, states_emb, attn_mask):
        inputs_emb = self.content_reflect(states_emb)
        inputs_emb = inputs_emb + self.positional_embeddings(
            torch.arange(self.state_size, device=states.device))
        seq = self.emb_dropout(inputs_emb)
        pad = torch.ne(states, 0).float().unsqueeze(-1)
        seq = seq * pad
        out = self.mh_attn(self.ln_1(seq), seq, seq, attn_mask)
        out = self.feed_forward(self.ln_2(out)) * pad
        return self.ln_3(out)

    def cacu_h(self, states, states_emb, len_states, p, attn_mask):
        """학습용 히스토리 인코딩 + CFG dropout. -> (h [B,H], ff_out [B,L,H])"""
        h, ff_out = self.predict(states, states_emb, len_states, attn_mask)
        keep = (torch.rand(h.size(0), 1, device=h.device) >= p).float()
        h = h * keep + self.none_embedding.weight * (1 - keep)
        return h, ff_out

    def predict(self, states, states_emb, len_states, attn_mask):
        ff_out = self._encode_seq(states, states_emb, attn_mask)
        idx = (len_states - 1).clamp(min=0)
        h = ff_out[torch.arange(ff_out.size(0), device=ff_out.device), idx]
        return h, ff_out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.w_2(F.relu(self.w_1(x.transpose(1, 2)))).transpose(1, 2)
        return self.layer_norm(self.dropout(out) + residual)


class Attention(nn.Module):
    def __init__(self, attention_dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.matmul(q, k.transpose(-2, -1))
        if scale is not None:
            attention = attention * scale
        if attn_mask is not None:
            if attention.shape[2] == attention.shape[3]:               # self attention
                m = attn_mask.unsqueeze(2) & attn_mask.unsqueeze(1)    # [B, N, N]
                attention = attention.masked_fill(~m.unsqueeze(1), float(-1e20))
            else:
                attention = attention.masked_fill(~attn_mask.unsqueeze(1).unsqueeze(1), float(-1e20))
        attention = self.dropout(self.softmax(attention))
        return torch.matmul(attention, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=128, out_dim=128, num_heads=8, dropout=0.5):
        super().__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.dot_product_attention = Attention(dropout)
        self.linear_final = nn.Linear(model_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        B1, N1, C1 = query.shape
        B2, N2, _ = key.shape
        residual = query
        key = self.linear_k(key).reshape(B2, N2, self.num_heads, self.dim_per_head).permute(0, 2, 1, 3)
        value = self.linear_v(value).reshape(B2, N2, self.num_heads, self.dim_per_head).permute(0, 2, 1, 3)
        query = self.linear_q(query).reshape(B1, N1, self.num_heads, self.dim_per_head).permute(0, 2, 1, 3)
        att = self.dot_product_attention(query, key, value, self.dim_per_head ** -0.5, attn_mask)
        att = att.transpose(1, 2).reshape(B1, N1, C1)
        out = self.dropout(self.linear_final(att))
        return self.layer_norm(residual + out)


class class_token_pos_embed(nn.Module):
    def __init__(self, embed_dim, num_tokens):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat((cls_token, x), dim=1) + self.pos_embed


# ----------------------------------------------------------------------------- IB classifier (Phase 1)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, latent_dim))

    def forward(self, x):
        return self.encoder(x)


class ModelWithEmbeddingIB(nn.Module):
    """정보 병목 기반 진위 분리기. label_embedding = {e_real(0), e_fake(1)}."""

    def __init__(self, input_dim, hidden_dim, bottleneck_dim, num_classes=2, logit_scale=1.5, dropout_rate=0.3):
        super().__init__()
        self.logit_scale = logit_scale
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        self.encoder_r = Encoder(input_dim, hidden_dim, bottleneck_dim)
        self.encoder_i = Encoder(input_dim, hidden_dim, bottleneck_dim)
        self.label_embedding = nn.Embedding(num_classes, input_dim)
        self.decoder = nn.Sequential(
            nn.Linear(2 * bottleneck_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim))
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(),
            nn.Dropout(dropout_rate), nn.Linear(hidden_dim, num_classes))

    @staticmethod
    def cost_matrix_cosine(x, y, eps=1e-5):
        cos = F.normalize(x, dim=-1, eps=eps) @ F.normalize(y, dim=-1, eps=eps).T
        return (1 - cos).unsqueeze(0)

    @staticmethod
    def trace(x):
        b, m, n = x.size()
        mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
        return x.masked_select(mask).view(b, n).sum(-1)

    @torch.no_grad()
    def ipot(self, C, beta, iteration, k, eps=1e-8):
        b, m, n = C.size()
        sigma = torch.ones(b, m, dtype=C.dtype, device=C.device)
        T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
        A = torch.exp(-C.transpose(1, 2) / beta)
        for _ in range(iteration):
            Q = A * T
            sigma = sigma.view(b, m, 1)
            for _ in range(k):
                delta = 1 / (Q.matmul(sigma).view(b, 1, n) + eps)
                sigma = 1 / (delta.matmul(Q) + eps)
            T = delta.view(b, n, 1) * Q * sigma
        return T

    def optimal_transport_dist(self, z_label, z_r, beta=0.5, iteration=50, k=1):
        cost = self.cost_matrix_cosine(z_label, z_r)
        T = self.ipot(cost.detach(), beta, iteration, k)
        return self.trace(cost.matmul(T.detach())).mean()

    def contrastive_loss_per_label(self, z_label, z_r, z_i, labels):
        total = 0.0
        uniq = torch.unique(labels)
        for label in uniq:
            idx = (labels == label).nonzero().view(-1)
            pos = z_r[idx].unsqueeze(1)
            neg = z_i[idx].unsqueeze(1)
            samples = F.normalize(torch.cat([pos, neg], dim=1), dim=-1)
            zl = F.normalize(z_label[idx], dim=-1)
            logits = self.logit_scale * F.cosine_similarity(zl.unsqueeze(1), samples, dim=2)
            bz, K, _ = pos.shape
            target = torch.cat([torch.ones(bz, K), torch.zeros(bz, K)], dim=1).to(zl.device)
            logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            total = total + self.kl_div_loss(logits, target)
        return total / len(uniq)

    def forward(self, x, labels):
        z_r = self.encoder_r(x)
        z_i = self.encoder_i(x)
        z_label = self.encoder_r(self.label_embedding(labels))
        proto = self.encoder_r(self.label_embedding(torch.arange(self.label_embedding.num_embeddings, device=x.device)))
        ot_distance = self.optimal_transport_dist(proto, z_r)
        contrastive = self.contrastive_loss_per_label(z_label, z_r, z_i, labels)

        x_recon, x_ori = [], []
        for label in torch.unique(labels):
            idx = (labels == label).nonzero().view(-1)
            z_r_shuf = z_r[idx[torch.randperm(idx.size(0), device=idx.device)]]
            x_recon.append(self.decoder(torch.cat([z_r_shuf, z_i[idx]], dim=1)))
            x_ori.append(x[idx])
        x_recon = torch.cat(x_recon, 0)
        x_ori = torch.cat(x_ori, 0)
        logits = self.classifier(z_r)
        return x_recon, z_label, z_r, z_i, ot_distance, x_ori, contrastive, logits


# ---- 벤치 래퍼

class PRISM(BaseModel):
    num_stages = 2

    def __init__(self, data, cfg, device):
        super().__init__(data, cfg, device)
        emb = self.full_emb()                                           # [N+1, 1536]
        input_dim = emb.size(1)
        assert input_dim == cfg["input_dim"], f"input_dim {cfg['input_dim']} != emb {input_dim}"
        self.register_buffer("labels_long", data["labels"].long())      # [N+1]

        self.classifier = ModelWithEmbeddingIB(input_dim, cfg["hidden_dim"], cfg["bottleneck_dim"], 2)
        self.diffusion = Diffusion(
            emb, input_dim=input_dim, hidden_size=cfg["hidden_dim"], timesteps=cfg["timesteps"],
            beta_start=cfg["beta_start"], beta_end=cfg["beta_end"],
            hyper_w=cfg["w"], max_len=cfg["max_len"], p=cfg["p"],
            dropout=cfg.get("dropout_rate", 0.1), num_heads=cfg.get("num_heads", 4))
        self.diffusion.eval_seed = cfg["seed"]   # 평가 노이즈 고정 (sample 재현성)

        # stage 0 용 뉴스 순회 상태
        self._gen = torch.Generator().manual_seed(cfg["seed"])
        self._perm = None
        self._pos = 0

    # ------------------------------------------------------------------ stage 제어
    def loader_overrides(self, stage):
        if stage == 0:   # 1 epoch = 뉴스 1회 순회 (배치 수 = ceil(N / batch_size))
            return {"batch_size": self.cfg["batch_size"], "max_samples": self.num_news}
        return {}

    def eval_enabled(self, stage):
        return stage == 1

    def configure_optimizer(self, stage):
        wd = self.cfg.get("weight_decay", 0.0)
        if stage == 0:
            return torch.optim.Adam(self.classifier.parameters(), lr=self.cfg["lr_cls"], weight_decay=wd)
        opt = torch.optim.Adam(self.diffusion.parameters(), lr=self.cfg["lr"], weight_decay=wd)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.cfg["decay_step"], gamma=self.cfg["gamma"])
        return opt, sched

    def on_stage_end(self, stage):
        if stage == 0:
            for p in self.classifier.parameters():
                p.requires_grad = False
            self.classifier.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode and not any(p.requires_grad for p in self.classifier.parameters()):
            self.classifier.eval()         # 동결 후에는 항상 eval
        return self

    # ------------------------------------------------------------------ stage 0: 분류기
    def _next_news_chunk(self):
        bs = self.cfg["batch_size"]
        if self._perm is None or self._pos >= len(self._perm):
            self._perm = torch.randperm(self.num_news, generator=self._gen) + 1
            self._pos = 0
        idx = self._perm[self._pos:self._pos + bs]
        self._pos += bs
        if idx.numel() < 2:                # BatchNorm 은 배치 1 불가
            self._perm = None
            return self._next_news_chunk()
        return idx.to(self.device)

    def _cls_loss(self):
        idx = self._next_news_chunk()
        x = self.diffusion.news_emb[idx]
        y = self.labels_long[idx]
        x_recon, _, _, _, ot, x_ori, contrastive, logits = self.classifier(x, y)
        recon = F.mse_loss(x_recon, x_ori)
        cls = F.cross_entropy(logits, y)
        loss = cls + self.cfg["phi"] * ot + self.cfg["tau"] * contrastive + self.cfg["lambda_r"] * recon
        acc = (logits.argmax(1) == y).float().mean()
        return loss, {"cls": cls.item(), "ot": ot.item(), "con": float(contrastive.detach()) if torch.is_tensor(contrastive) else float(contrastive), "rec": recon.item(), "acc": acc.item()}

    # ------------------------------------------------------------------ stage 1: 디퓨전
    def _anchor(self, label):
        return self.classifier.label_embedding(torch.tensor([label], device=self.device))

    def _diff_loss(self, batch):
        seq, mask, target = batch["ctx"], batch["mask"], batch["target"]
        loss_d, scores = self.diffusion.p_losses(seq, mask, target, self._anchor(0))
        loss_r = F.cross_entropy(scores, target)
        loss = loss_d + self.cfg["diff_cof"] * loss_r
        return loss, {"L_d": loss_d.item(), "L_rec": loss_r.item()}

    def compute_loss(self, batch, stage=0):
        if stage == 0:
            return self._cls_loss()
        return self._diff_loss(batch)

    # ------------------------------------------------------------------ 평가
    @torch.no_grad()
    def score(self, batch):
        _, scores = self.diffusion.sample(batch["ctx"], batch["mask"], self._anchor(0), self._anchor(1))
        return scores
