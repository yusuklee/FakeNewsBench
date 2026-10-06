# FakeNewsBench

FAKE NEWS DETECTION MODELS (Rec4Mit, HDInt, PRISM)


```
python prepare_data.py --dataset all   (all, gossip, pol, pheme, ced, mcfend 선택)
python main.py --model rec4mit --dataset gossip    학습 + 테스트
python main.py --model hdint   --dataset gossip
python main.py --model prism   --dataset gossip
python main.py --model prism   --dataset gossip --test_only   학습은 이미 완료했고 test만 하고싶을떄
```

## 폴더구조

```
FakeNewsBench/
├── data/
│   ├── {dataset}.csv       
│   └── processed/{dataset}/           데이터 전처리 결과 저장되는곳
├── prepare_data.py          데이터 저장과 전처리
├── datasets.py              임베딩, 인스턴스를 모델이 학습할 수 있는 형태로 제공
├── models/
│   ├── base.py              모델 공통 으로 부모로 상속
│   ├── rec4mit.py           WWW'22  Rec4Mit
│   ├── hdint.py             KDD'24  HDInt
│   └── prism.py             SIGIR'25 PRISM
├── trainer.py               학습
├── evaluate.py              평가
├── config.py                파라미터 설정하는곳
├── main.py                  학습 + 테스트
├── checkpoints/{model}/{dataset}/best.pt, test_results.json
└── logs/{model}/{dataset}/{timestamp}.log
```

## 사용 데이터

https://github.com/yusuklee/FakeNewsBench/releases/tag/v0.1.0


## 전처리 (`prepare_data.py`)

1. NEWS1:[USER1, USER2,..] -> USER1: [NEWS1, NEWS2, .. ]   로 변환
2. 인스턴스 길이 5
3. 인스턴스 분할은 인스턴스들을 모은후  **target 시각순** 정렬하여 분할.  ->  80% train / 10% val / 10% test.
4. text -> tokens -> emb     save tokens , embs 
   emb tools->  영어 `bert-base-uncased`, 중국어 `bert-base-chinese`.

단어 벡터 파일은 직접 받아서 저장소 폴더(`FakeNewsBench/`)에 둔다 (git 미포함).

- 영어: `glove.840B.300d.zip` — https://nlp.stanford.edu/data/glove.840B.300d.zip
- 중국어: `sgns.merge.word.bz2` — https://github.com/Embedding/Chinese-Word-Vectors 의 Mixed-large / Word

결과 위치 `data/processed/{dataset}/`

| 파일 | 내용 | 사용 모델 |
|---|---|---|
| news_emb.pt | {"title": [N+1,768], "description": [N+1,768]}, row 0 = 패딩 | Rec4Mit, PRISM |
| tokens.pt | title/desc input_ids, attention_mask | HDInt stage1 (BERT fine-tune) |
| category.pt | 뉴스별 카테고리 번호 [N+1]. 제목 TF-IDF → K-means (K = min(300, 뉴스 수 // 100)). CSV에 `category` 컬럼이 있으면(pheme, PHEME 사건 9개) 그 값을 사용 | 아직 없음 (CAUM, FUM 용) |
| subcategory.pt | 뉴스별 서브카테고리 번호 [N+1]. 카테고리 안에서 K-means 한 번 더 (K = 카테고리 뉴스 수 // 33, 최소 1) | 아직 없음 (FUM 용) |
| entity.pt | 뉴스별 엔티티 번호 [N+1, 5]. 제목 NER (영어 `dslim/bert-base-NER`, 중국어 `uer/roberta-base-finetuned-cluener2020-chinese`), 0 = 없음 | 아직 없음 (CAUM, FUM 용) |
| words.pt | 단어 번호 {"title": [N+1,30], "description": [N+1,50]} + 단어 벡터 표 {"emb": [V+1,300]}. 영어 GloVe, 중국어 Chinese Word Vectors | 아직 없음 (NRMS, CAUM, FUM, RobustSentiRec 용) |
| sentiment.pt | 뉴스별 제목 감성 점수 [N+1], -1(부정) ~ +1(긍정) | 아직 없음 (RobustSentiRec 용) |
| train/val/test.json | `[[ctx_ids], target_id, user_idx, target_time]` | 전체 |
| meta.json | 뉴스/유저/인스턴스 수, unseen 비율 | 전체 |  -> 그냥 설명하는 기능



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



## 평가 (`evaluate.py`)

PRISM 논문 지표, 전체 뉴스 라이브러리 full-ranking, K = 5, 10, 20.

| 지표 | 의미 |
|---|---|
| HR@K | 정답이 top-K 안에 있는 비율 |
| NDCG@K | 1 / log2(rank+1) |
| RT@K | top-K 중 진짜 뉴스 비율 |
| FNSR@K | 1 − Σ(fake 순위 가중치) / Σ(1..K). 상위 가짜일수록 큰 패널티 |
| F1@K | 2 · HR · FNSR / (HR + FNSR) |
| 1-R | 1위가 진짜 뉴스인 비율 |

best 체크포인트는 val HR@5 기준 (`select_metric`).

## 설정

`config.py` 하나. COMMON (max_len 5, num_neg 4, batch 64, seed 42 ...) 위에 MODEL_DEFAULTS[model]을 덮는다. 바꾸려면 파일을 고친다.

## 모델 차이

각 모델 파일 상단 docstring에 원본 대비 바뀐 점을 적어 두었다. 공통으로 바뀐 것:

- 히스토리 길이 5 (Rec4Mit·HDInt 4, PRISM 10 -> 5)
- 분할: 단일 시간순 8:1:1 (Rec4Mit 10-fold, HDInt 랜덤, PRISM 유저 단위 → 통일)
- 텍스트: title + description만 (PRISM 본문 text 없음)
-   HDInt용 KeyBERT 키워드 3개, 정치성향 라벨, PRISM의 P_c/P_u 시간 분리.

| 모델 | 학습 | 원본 대비 |
|---|---|---|
| Rec4Mit | 1단계, 15 epoch | 메타 임베딩은 공통 BERT CLS를 필드별 정규화(`normalize_emb`). 정답이 real인 인스턴스만 예측 손실(`train_real_target_only`). 평가 시 자체 분류기로 fake 후보 제외(`filter_fake`). 적대 손실 1/BCE 가 가끔 폭발하는 건 원본 식(Eq 11) 그대로 |
| HDInt | 2단계: BERT fine-tune 5 → frozen 15 | 키워드 없음 → title 임베딩 ×3 으로 대체. 정치성향 없음 → 전부 중립(1). 둘 다 TODO |
| PRISM | 2단계: IB 분류기 30 → 디퓨전 120 | P_c/P_u 분리 없음 (분류기가 전체 라이브러리로 학습, TODO). 입력 1536-d. 평가 노이즈 시드 고정 |
