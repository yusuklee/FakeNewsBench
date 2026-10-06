"""HDInt (KDD 2024) — 벤치 래퍼. 2단계 학습.

  stage 0  BERT fine-tune : 배치 안 뉴스 토큰을 BERT 로 실시간 인코딩 (상위 N층만 학습) + HDInt 헤드 동시 학습.
                            stage1_max_samples 개 인스턴스, 작은 배치 + grad accum.
  on_stage_end(0)         : fine-tuned BERT 로 전체 뉴스 임베딩 추출 -> 버퍼 저장, 헤드 재초기화, BERT 동결.
  stage 1  frozen         : 저장된 임베딩으로 HDInt 헤드만 학습.

입력 조립 (논문 4.1.1 / 4.2.1)
  content = [title CLS ; description CLS]               -> 1536
  keyword = [k1 CLS ; k2 CLS ; k3 CLS]                   -> 2304

TODO (사용자 결정으로 보류된 것 — 전처리에 컬럼이 생기면 바꿀 것)
  * 키워드 3개가 processed 데이터에 없다. 임시로 title CLS 를 3번 복제해서 keyword 입력으로 쓴다
    (shape 2304 유지, high-level interest learner 는 그대로 동작).  -> data["tokens"]["k{1,2,3}_ids/mask"]
    가 생기면 _encode_news() 의 fallback 분기만 제거하면 된다.
  * 정치성향 라벨이 없다. 전부 1(중립) 로 둔다. PolarityDisentangler 는 구조상 그대로 있고 손실도 계산된다
    (단일 클래스라 label loss 는 바로 0 근처로 수렴). -> self.pol_lbl 버퍼를 실제 라벨로 채우면 된다.

원본 대비 기타 차이
  * grad clip(1.0) 과 ReduceLROnPlateau 는 공통 trainer 에 없어 생략.
  * stage 0 는 CUDA 에서 bf16 autocast 사용 (원본은 fp16 + GradScaler).
  * 유저 임베딩: 벤치 user idx(0..U-1) 를 직접 사용 (원본은 1-based + padding).
"""

import torch
import torch.nn as nn

from models.base import BaseModel
from models.hdint_parts import HDIntCore

CONTENT_DIM = 1536
KEYWORD_DIM = 2304


class HDInt(BaseModel):
    num_stages = 2

    def __init__(self, data, cfg, device):
        super().__init__(data, cfg, device)
        from transformers import BertModel

        bert_name = data["meta"].get("bert") or "bert-base-uncased"
        if bert_name == "none":
            bert_name = "bert-base-uncased"
        self.bert = BertModel.from_pretrained(bert_name)
        self._freeze_bert(int(cfg.get("stage1_unfreeze_layers", 3)))

        tok = data.get("tokens")
        if tok is None:
            raise ValueError("HDInt needs data['tokens'] (tokens.pt). Run prepare_data.py first.")
        self.tokens = {k: v for k, v in tok.items()}                  # CPU
        self.has_keywords = all(f"k{i}_ids" in tok for i in (1, 2, 3))

        self.core = self._new_core()

        N = self.num_news + 1
        self.register_buffer("content_emb", torch.zeros(N, CONTENT_DIM))
        self.register_buffer("keyword_emb", torch.zeros(N, KEYWORD_DIM))
        self.register_buffer("emb_ready", torch.tensor(False))
        self.register_buffer("ver_lbl", data["labels"].long())
        self.register_buffer("pol_lbl", torch.ones(N, dtype=torch.long))  # TODO: 실제 정치성향 라벨

        self._step = 0
        self._cache_step = -1
        self._cache = None       # (content_all, keyword_all) stage 0 평가용
        self._cand_cache = None  # (clean, keyword_proj)

    # ------------------------------------------------------------------ 구성
    def _new_core(self):
        c = self.cfg
        return HDIntCore(CONTENT_DIM, KEYWORD_DIM, c["embed_dim"], c["num_heads"],
                         c.get("num_polarity_classes", 3), c.get("num_transformer_layers", 1),
                         c.get("dropout", 0.1), num_users=self.num_users,
                         user_embed_dim=c.get("user_embed_dim", 32)).to(self.device)

    def _freeze_bert(self, unfreeze_layers: int):
        for p in self.bert.parameters():
            p.requires_grad = False
        if unfreeze_layers > 0:
            for layer in self.bert.encoder.layer[-unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

    # ------------------------------------------------------------------ BERT 인코딩
    def _cls(self, ids, mask):
        empty = mask.sum(1) == 0
        safe = mask.clone()
        if empty.any():
            safe[empty, 0] = 1
        cls = self.bert(input_ids=ids, attention_mask=safe).last_hidden_state[:, 0, :]
        return cls * (~empty).float().unsqueeze(1)

    def _encode_news(self, idx: torch.Tensor):
        """뉴스 idx [M] (CPU) -> content [M,1536], keyword [M,2304] (device)."""
        t = {k: v[idx].to(self.device) for k, v in self.tokens.items()}
        title = self._cls(t["title_ids"], t["title_mask"])
        desc = self._cls(t["desc_ids"], t["desc_mask"])
        content = torch.cat([title, desc], dim=1)
        if self.has_keywords:
            ks = [self._cls(t[f"k{i}_ids"], t[f"k{i}_mask"]) for i in (1, 2, 3)]
        else:
            ks = [title, title, title]                                  # TODO fallback (키워드 없음)
        return content, torch.cat(ks, dim=1)

    @torch.no_grad()
    def _encode_all_news(self, batch_size=256):
        was_training = self.training
        self.bert.eval()
        cs, ks = [], []
        for s in range(0, self.num_news + 1, batch_size):
            idx = torch.arange(s, min(s + batch_size, self.num_news + 1))
            c, k = self._encode_news(idx)
            cs.append(c.float())
            ks.append(k.float())
        if was_training:
            self.bert.train()
        return torch.cat(cs), torch.cat(ks)

    # ------------------------------------------------------------------ trainer 훅
    def loader_overrides(self, stage):
        if stage == 0:
            return {"batch_size": self.cfg["stage1_batch_size"], "grad_accum": self.cfg["stage1_grad_accum"],
                    "max_samples": self.cfg["stage1_max_samples"]}
        return {}

    def configure_optimizer(self, stage):
        if stage == 0:
            bert_p = [p for p in self.bert.parameters() if p.requires_grad]
            return torch.optim.AdamW([{"params": bert_p, "lr": self.cfg["stage1_bert_lr"]},
                                      {"params": self.core.parameters(), "lr": self.cfg["lr"]}],
                                     weight_decay=self.cfg.get("stage1_weight_decay", 0.01))
        return torch.optim.Adam(self.core.parameters(), lr=self.cfg["lr"])

    def on_stage_end(self, stage):
        if stage != 0:
            return
        content, keyword = self._encode_all_news()
        self.content_emb.copy_(content)
        self.keyword_emb.copy_(keyword)
        self.emb_ready.fill_(True)
        self.core = self._new_core()                                    # 원본: stage 2 헤드 새로 시작
        for p in self.bert.parameters():
            p.requires_grad = False
        self._cache = None
        self._cache_step = -1

    # ------------------------------------------------------------------ 손실
    @staticmethod
    def _bce(pos, neg):
        return -torch.log(pos + 1e-8).mean() - torch.log(1.0 - neg + 1e-8).mean()

    def _loss_from_emb(self, batch, ctx_c, ctx_k, tgt_c, tgt_k, neg_c, neg_k):
        ctx, mask = batch["ctx"], batch["mask"]
        pad = ~mask
        ctx_pol, ctx_ver = self.pol_lbl[ctx], self.ver_lbl[ctx]
        tgt, neg, uid = batch["target"], batch["neg"], batch["user"]
        K = neg.size(1)

        pos, pol_ctx, pol_p, ver_ctx, ver_p = self.core(ctx_c, ctx_k, tgt_c, tgt_k, pad, ctx_pol, ctx_ver,
                                                        self.pol_lbl[tgt], self.ver_lbl[tgt], uid)
        negs, pol_n, ver_n = [], 0.0, 0.0
        for k in range(K):
            nid = neg[:, k]
            ns, _, p_n, _, v_n = self.core(ctx_c, ctx_k, neg_c[:, k], neg_k[:, k], pad, ctx_pol, ctx_ver,
                                           self.pol_lbl[nid], self.ver_lbl[nid], uid)
            negs.append(ns)
            pol_n = pol_n + p_n
            ver_n = ver_n + v_n
        neg_s = torch.stack(negs, 1)
        bce = self._bce(pos, neg_s)
        pol = pol_ctx + pol_p + pol_n / K
        ver = ver_ctx + ver_p + ver_n / K
        loss = self.cfg["lambda"] * bce + self.cfg["gamma"] * pol + ver
        return loss, {"bce": bce.item(), "pol": float(pol.detach()) if torch.is_tensor(pol) else float(pol),
                      "ver": float(ver.detach()) if torch.is_tensor(ver) else float(ver)}

    def compute_loss(self, batch, stage=0):
        self._step += 1
        ctx, tgt, neg = batch["ctx"], batch["target"], batch["neg"]
        B, S = ctx.shape
        K = neg.size(1)
        if stage == 0:
            all_ids = torch.cat([ctx.reshape(-1), tgt, neg.reshape(-1)])
            uniq, inv = torch.unique(all_ids, return_inverse=True)
            use_amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                c_u, k_u = self._encode_news(uniq.cpu())
                c_u, k_u = c_u.float(), k_u.float()
                ctx_c = c_u[inv[:B * S]].reshape(B, S, -1)
                ctx_k = k_u[inv[:B * S]].reshape(B, S, -1)
                tgt_c, tgt_k = c_u[inv[B * S:B * S + B]], k_u[inv[B * S:B * S + B]]
                neg_c = c_u[inv[B * S + B:]].reshape(B, K, -1)
                neg_k = k_u[inv[B * S + B:]].reshape(B, K, -1)
                return self._loss_from_emb(batch, ctx_c, ctx_k, tgt_c, tgt_k, neg_c, neg_k)
        ce, ke = self.content_emb, self.keyword_emb
        return self._loss_from_emb(batch, ce[ctx], ke[ctx], ce[tgt], ke[tgt], ce[neg], ke[neg])

    # ------------------------------------------------------------------ 평가
    def _emb_tables(self):
        """현재 단계에 맞는 (content, keyword) 전체 테이블. stage 0 에서는 현재 BERT 로 인코딩(캐시)."""
        if bool(self.emb_ready):
            return self.content_emb, self.keyword_emb
        if self._cache is None or self._cache_step != self._step:
            self._cache = self._encode_all_news()
            self._cache_step = self._step
            self._cand_cache = None
        return self._cache

    @torch.no_grad()
    def _candidates(self, content, keyword):
        if self._cand_cache is None or self._cand_cache[0] != self._step:
            clean, _, _ = self.core.disentangler(self.core.proj_content(content))
            self._cand_cache = (self._step, (clean, self.core.keyword_proj(keyword)))
        return self._cand_cache[1]

    @torch.no_grad()
    def score(self, batch):
        content, keyword = self._emb_tables()
        cand = self._candidates(content, keyword)
        ctx, mask = batch["ctx"], batch["mask"]
        return self.core.score_batch(content[ctx], keyword[ctx], ~mask, batch["user"], cand)

    @torch.no_grad()
    def candidate_mask(self):
        if not self.cfg.get("filter_fake", False):
            return None
        content, _ = self._emb_tables()
        prob = self.core.disentangler.predict_veracity(self.core.proj_content(content))
        m = prob >= self.cfg.get("fake_threshold", 0.5)
        m[0] = False
        return m

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._cand_cache = None
        return self

    def load_state_dict(self, *args, **kwargs):
        r = super().load_state_dict(*args, **kwargs)
        self._cache, self._cand_cache, self._cache_step = None, None, -1   # 가중치 바뀌면 캐시 무효화
        return r
