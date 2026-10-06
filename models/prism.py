"""PRISM (SIGIR'25) — 정보 병목 진위 분리기 + 조건부 디퓨전 추천기 (CFG).

stage 0 : IB 분류기 (ModelWithEmbeddingIB) 를 뉴스 라이브러리 전체로 학습 → e_real / e_fake 앵커 확보 → 동결
stage 1 : 디퓨전 추천기 학습. L = L_d(노이즈 MSE) + diff_cof * CE(cosine 스코어, target)
score   : CFG 샘플링 (w, e_real / e_fake) 으로 타깃 임베딩 생성 → 전체 뉴스와 코사인

원 논문/재현 코드와 다른 점 (벤치 통일 결정)
  - P_c / P_u 시간 분리 미적용: 분류기도 추천 후보와 같은 뉴스 전체로 학습 (P_c ∩ P_u = ∅ 미보장)
  - 뉴스 임베딩 = title 768 + description 768 (1536). 본문 text 없음
  - 히스토리 길이 cfg["max_len"] (=5)
  - stage 0 의 1 epoch = 뉴스 라이브러리 1회 순회 (트레이너의 인스턴스 배치는 무시)
  - best 선택은 공통 select_metric (HR@5). 원 코드의 복합 지표 미사용
"""

import torch
import torch.nn.functional as F

from models.base import BaseModel
from models.prism_parts import Diffusion, ModelWithEmbeddingIB


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
            beta_start=cfg["beta_start"], beta_end=cfg["beta_end"], beta_sche=cfg.get("beta_sche", "linear"),
            hyper_w=cfg["w"], fusion_mode=cfg["fusion_mode"], max_len=cfg["max_len"], p=cfg["p"],
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
