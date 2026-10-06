# FakeNewsBench

가짜뉴스 완화 뉴스추천 모델(Rec4Mit, HDInt, PRISM)을 **같은 데이터, 같은 전처리, 같은 평가**로 비교하는 벤치마크.
NNR(https://github.com/Veason-silverbullet/NNR) 처럼 전처리·Dataset·학습 루프·평가를 공유하고 모델만 `--model`로 바꿔 돌린다.

```
python prepare_data.py --dataset all                 # 전처리 1회
python main.py --model rec4mit --dataset gossip      # 학습 + 테스트
python main.py --model hdint   --dataset gossip
python main.py --model prism   --dataset gossip
python main.py --model prism   --dataset gossip --test_only   # 저장된 best.pt 평가
```

## 폴더

```
FakeNewsBench/
├── data/
│   ├── raw/                 입력 CSV (gossip, pol, pheme, ced, mcfend) — git 미포함, releases 참조
│   └── processed/{dataset}/ prepare_data.py 산출물 — git 미포함
├── prepare_data.py          전처리 (1회)
├── datasets.py              processed 로드 + 인스턴스 -> 배치 텐서 (모델 공통, 분기 없음)
├── models/
│   ├── base.py              공통 인터페이스 (compute_loss / score / stage hooks)
│   ├── rec4mit.py           WWW'22  Rec4Mit
│   ├── hdint.py             KDD'24  HDInt
│   └── prism.py             SIGIR'25 PRISM
├── trainer.py               공통 학습 루프 (stage, val 평가, best 저장)
├── evaluate.py              PRISM 논문 지표, full-ranking
├── config.py                공통 하이퍼파라미터 + 모델별 기본값
├── main.py                  진입점
├── checkpoints/{model}/{dataset}/best.pt, test_results.json
└── logs/{model}/{dataset}/{timestamp}.log
```

## 입력 데이터

`data/raw/{dataset}.csv`, 컬럼은 5개 데이터셋 모두 동일.

| 컬럼 | 내용 |
|---|---|
| news_id | 뉴스 ID |
| title | 제목 |
| description | 요약. 비어 있거나 title과 같으면 없는 것으로 처리 |
| label | 0 = real, 1 = fake |
| user_ids | 이 뉴스를 공유한 유저 ID 리스트 |
| user_times | 각 유저의 공유 시각 (unix) 리스트 |

| 데이터셋 | 뉴스 | 출처 |
|---|---|---|
| gossip | 17,527 | FakeNewsNet GossipCop (UPFD + DECOR) |
| pol | 599 | FakeNewsNet PolitiFact (UPFD + DECOR) |
| pheme | 5,728 | PHEME |
| ced | 3,387 | CED (중국어) |
| mcfend | 6,808 | MCFEND (중국어) |

## 전처리 (`prepare_data.py`)

모든 모델이 아래 산출물을 그대로 쓴다. 분할 비율 등은 상수로 고정 (CLI로 못 바꿈).

1. `user_ids`/`user_times` → 유저별 시간순 뉴스 시퀀스. 같은 뉴스 재등장은 첫 번째만.
2. 슬라이딩 윈도우 인스턴스: context = 직전 최대 **5개**, target = 다음 뉴스.
3. 인스턴스 전체를 **target 시각순** 정렬 → 앞 80% train / 10% val / 10% test.
   (랜덤 셔플이 아니라 시간순이라 test에 train에 없던 뉴스가 생김 = PRISM 논문 Table 1의 Unseen)
4. BERT CLS 임베딩 (title 32토큰, description 128토큰, 각 768, 정규화 안 함) + 토큰 저장.
   영어 `bert-base-uncased`, 중국어 `bert-base-chinese`.

산출물 `data/processed/{dataset}/`

| 파일 | 내용 | 사용 모델 |
|---|---|---|
| news.csv | idx, news_id, title, description, label | 전체 |
| news_emb.pt | {"title": [N+1,768], "description": [N+1,768]}, row 0 = 패딩 | Rec4Mit, PRISM, HDInt stage2 |
| tokens.pt | title/desc input_ids, attention_mask | HDInt stage1 (BERT fine-tune) |
| train/val/test.json | `[[ctx_ids], target_id, user_idx, target_time]` | 전체 |
| meta.json | 뉴스/유저/인스턴스 수, unseen 비율 | 전체 |

아직 안 넣은 것 (추후): HDInt용 KeyBERT 키워드 3개, 정치성향 라벨, PRISM의 P_c/P_u 시간 분리.

## Dataset (`datasets.py`)

`BenchDataset` 하나. 인스턴스 1개 → 배치 텐서. 모델 분기 없음.

| 키 | 형태 | 내용 |
|---|---|---|
| ctx | Long [5] | 히스토리 뉴스 idx, 오른쪽 0 패딩 |
| mask | Bool [5] | 실제 뉴스 위치 |
| target | Long | 정답 뉴스 idx |
| user | Long | 유저 idx |
| neg | Long [4] | 네거티브 (real 2 + fake 2). 평가 시 0. PRISM은 무시 |
| ctx_label / target_label / neg_label | Float | 진위 라벨 |

임베딩·토큰 조회는 모델이 `data["emb"]`, `data["tokens"]`에서 직접 한다.

## 모델 인터페이스 (`models/base.py`)

```python
class MyModel(BaseModel):
    num_stages = 1                       # 다단계 학습이면 2
    def compute_loss(self, batch, stage) -> (loss, {"name": value})
    def score(self, batch) -> Tensor[B, N+1]   # 전체 뉴스 점수, full ranking
    # 선택: configure_optimizer(stage), on_stage_start/end(stage),
    #       eval_enabled(stage), loader_overrides(stage), candidate_mask()
```

`models/__init__.py`의 REGISTRY에 이름을 등록하면 `--model`로 호출된다.

## 평가 (`evaluate.py`)

PRISM 논문 지표, 전체 뉴스 라이브러리 full-ranking, K = 5, 10, 20.

| 지표 | 의미 |
|---|---|
| HR@K | 정답이 top-K 안에 있는 비율 |
| NDCG@K | 1 / log2(rank+1) |
| MRR@K | 1 / rank |
| RT@K | top-K 중 진짜 뉴스 비율 |
| WFNS@K | 1 − Σ(fake 순위 가중치) / Σ(1..K). 상위 가짜일수록 큰 패널티 |
| 1-R | 1위가 진짜 뉴스인 비율 |

best 체크포인트는 val HR@5 기준 (`select_metric`).

## 설정

`config.py`의 COMMON (max_len 5, num_neg 4, batch 64, seed 42 ...) 위에 MODEL_DEFAULTS[model]을 덮고,
CLI `--set key=value`로 다시 덮는다.

```
python main.py --model hdint --dataset pheme --set "epochs=[1,3]" lr=0.0005
```

## 모델별 메모

각 모델 파일 상단 docstring에 원본 대비 바뀐 점을 적어 두었다. 공통으로 바뀐 것:

- 히스토리 길이 5 (Rec4Mit·HDInt 4, PRISM 10 → 5)
- 분할: 단일 시간순 8:1:1 (Rec4Mit 10-fold, HDInt 랜덤, PRISM 유저 단위 → 통일)
- 텍스트: title + description만 (PRISM 본문 text 없음)
- 평가: 전부 full-ranking (HDInt 99 네거티브 샘플링 → full)

| 모델 | 학습 | 원본 대비 |
|---|---|---|
| Rec4Mit | 1단계, 15 epoch | 메타 임베딩은 공통 BERT CLS를 필드별 정규화(`normalize_emb`). 정답이 real인 인스턴스만 예측 손실(`train_real_target_only`). 평가 시 자체 분류기로 fake 후보 제외(`filter_fake`). 적대 손실 1/BCE 가 가끔 폭발하는 건 원본 식(Eq 11) 그대로 |
| HDInt | 2단계: BERT fine-tune 5 → frozen 15 | 키워드 없음 → title 임베딩 ×3 으로 대체. 정치성향 없음 → 전부 중립(1). 둘 다 TODO |
| PRISM | 2단계: IB 분류기 30 → 디퓨전 120 | P_c/P_u 분리 없음 (분류기가 전체 라이브러리로 학습, TODO). 입력 1536-d. 평가 노이즈 시드 고정 |
