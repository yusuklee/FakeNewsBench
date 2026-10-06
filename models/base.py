"""모든 모델의 공통 인터페이스. trainer / evaluate 는 이 메서드만 호출한다."""

import os

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    필수 구현
      compute_loss(batch, stage) -> (loss: Tensor, logs: dict[str, float])
      score(batch) -> FloatTensor [B, N+1]   전체 뉴스 점수 (높을수록 추천). 0번 열은 무시됨
    선택 구현
      num_stages                  다단계 학습 수 (기본 1)
      configure_optimizer(stage)  -> optimizer 또는 (optimizer, scheduler)
      on_stage_start / on_stage_end(stage)
      eval_enabled(stage)         이 stage 에서 val 평가를 할지 (PRISM phase1 = False)
      loader_overrides(stage)     {"batch_size":..., "max_samples":...} 학습 로더 덮어쓰기
      candidate_mask()            -> Bool [N+1] 평가 시 후보에서 제외할 뉴스 (True = 제외). 기본 = P_c 뉴스
    """
    num_stages = 1

    def __init__(self, data: dict, cfg: dict, device):
        super().__init__()
        self.data = data
        self.cfg = cfg
        self.device = device
        self.num_news = data["num_news"]
        self.num_users = data["num_users"]
        self.register_buffer("pc", data["pc"], persistent=False)   # [N+1] True = P_c 뉴스
        self._cache = {}

    # ---- 필수
    def compute_loss(self, batch: dict, stage: int = 0):
        raise NotImplementedError

    @torch.no_grad()
    def score(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    # ---- 선택
    def configure_optimizer(self, stage: int = 0):
        return torch.optim.Adam(self.parameters(), lr=self.cfg["lr"],
                                weight_decay=self.cfg.get("weight_decay", 0.0))

    def on_stage_start(self, stage: int):
        pass

    def on_stage_end(self, stage: int):
        pass

    def eval_enabled(self, stage: int) -> bool:
        return True

    def loader_overrides(self, stage: int) -> dict:
        return {}

    def candidate_mask(self):
        return self.pc

    def train(self, mode: bool = True):
        self._cache = {}  # 평가용 캐시(전체 뉴스 벡터 등)는 train/eval 이 바뀔 때 버린다
        return super().train(mode)

    # ---- 유틸
    def load_file(self, name: str):
        """processed/{dataset}/ 의 전처리 파일 (words.pt, category.pt, entity.pt ...)."""
        return torch.load(os.path.join(self.data["dir"], name))

    @torch.no_grad()
    def all_news_vecs(self, encode, chunk: int = 1024) -> torch.Tensor:
        """encode(idx [n]) -> [n, D] 를 전체 뉴스(0..N)에 대해 한 번만 계산해 캐시 (평가용)."""
        if "news" not in self._cache:
            idx = torch.arange(self.num_news + 1, device=self.device)
            self._cache["news"] = torch.cat([encode(i) for i in idx.split(chunk)])
        return self._cache["news"]

    @staticmethod
    def to_device(batch: dict, device) -> dict:
        return {k: v.to(device) for k, v in batch.items()}

    @staticmethod
    def left_pad(ctx: torch.Tensor, mask: torch.Tensor):
        """오른쪽 패딩 [a,b,c,0,0] -> 왼쪽 패딩 [0,0,a,b,c] (최근 뉴스가 마지막 위치)."""
        L = ctx.size(1)
        lens = mask.sum(1)
        idx = torch.arange(L, device=ctx.device).unsqueeze(0) - (L - lens).unsqueeze(1)
        valid = idx >= 0
        idx = idx.clamp(min=0)
        out = torch.gather(ctx, 1, idx) * valid
        return out, valid

    def full_emb(self, which=("title", "description")) -> torch.Tensor:
        """[N+1, 768*len(which)] 고정 임베딩 테이블 (concat)."""
        return torch.cat([self.data["emb"][w] for w in which], dim=-1)
