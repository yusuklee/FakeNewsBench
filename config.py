"""공통 하이퍼파라미터 + 모델별 기본값. MODEL_DEFAULTS[model] 이 COMMON 을 덮는다."""

import random

import numpy as np
import torch

COMMON = {
    # 데이터
    "max_len": 5,            # 히스토리 길이 (전처리와 동일해야 함)
    "num_neg": 4,            # 학습 네거티브 수 (real 2 + fake 2). PRISM은 미사용
    "batch_size": 64,
    "eval_batch_size": 256,
    "num_workers": 0,
    # 학습
    "epochs": 15,            # int 또는 stage별 list
    "lr": 1e-3,
    "weight_decay": 0.0,
    "seed": 42,
    "eval_every": 1,
    "select_metric": "HR@5", # best 체크포인트 선택 기준
    # 평가 (PRISM 논문 지표, full-ranking)
    "ks": [5, 10, 20],
    "exclude_history": False, # True면 히스토리 뉴스를 후보에서 제외
}

MODEL_DEFAULTS = {
    "rec4mit": {
        "epochs": 15,
        "lr": 1e-3,
        "k_events": 20,
        "v_dim": 256,
        "e_dim": 128,
        "user_dim": 128,
        "filter_fake": True,     # 평가 시 분류기 예측 fake를 후보에서 제외 (원 논문 방식)
        "fake_threshold": 0.5,
        "train_real_target_only": True,
        "normalize_emb": True,  # 원본: 정답이 real인 인스턴스만 학습 (예측 손실에만 적용)
    },
    "hdint": {
        "epochs": [5, 15],       # [stage1: BERT fine-tune, stage2: frozen HDInt]
        "lr": 3e-4,
        "stage1_bert_lr": 2e-5,
        "stage1_batch_size": 8,
        "stage1_grad_accum": 8,
        "stage1_max_samples": 15000,
        "stage1_unfreeze_layers": 3,
        "embed_dim": 256,
        "num_heads": 4,
        "num_transformer_layers": 1,
        "dropout": 0.1,
        "lambda": 10.0,
        "gamma": 3.0,
        "num_polarity_classes": 3,
        "filter_fake": False,
        "fake_threshold": 0.5,
    },
    "prism": {
        "epochs": [30, 120],     # [phase1: IB 분류기, phase2: 디퓨전]
        "batch_size": 256,
        "lr": 1e-3,
        "lr_cls": 1e-3,
        "input_dim": 1536,       # title 768 + description 768
        "hidden_dim": 128,
        "bottleneck_dim": 128,
        "timesteps": 500,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "w": 21.0,
        "p": 0.1,
        "diff_cof": 0.1,
        "phi": 0.8,
        "tau": 0.4,
        "lambda_r": 1.0,
        "decay_step": 100,
        "gamma": 0.1,
        "eval_every": 5,
    },
}

# 데이터셋별 BERT (전처리용)
BERT_BY_DATASET = {
    "gossip": "bert-base-uncased",
    "pol": "bert-base-uncased",
    "pheme": "bert-base-uncased",
    "ced": "bert-base-chinese",
    "mcfend": "bert-base-chinese",
}
RAW_URL = "https://github.com/yusuklee/FakeNewsBench/releases/download/v0.1.0/"
RAW_FILES = {
    "gossip": "gossip.csv",
    "pol": "pol.csv",
    "pheme": "pheme.csv",
    "ced": "ced.csv",
    "mcfend": "mcfend.csv",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(model: str) -> dict:
    cfg = dict(COMMON)
    cfg.update(MODEL_DEFAULTS.get(model, {}))
    set_seed(cfg["seed"])
    return cfg
