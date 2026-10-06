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
│   ├── prism.py             SIGIR'25 PRISM
│   ├── nrms.py              EMNLP'19 NRMS
│   ├── miner.py             ACL'22  MINER
│   ├── caum.py              SIGIR'22 CAUM
│   ├── fum.py               SIGIR'22 FUM
│   └── robust_sentirec.py   RobustSentiRec (Sertkan et al., 2022)
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

0. 뉴스를 시각(가장 먼저 공유된 시각)순으로 2:8 분리. 앞 20% = P_c, 뒤 80% = P_u.  -> 다른 뉴스들은 80% 가지고만 인스턴스 만듬
   
1. NEWS1:[USER1, USER2,..] -> USER1: [NEWS1, NEWS2, .. ]   로 변환
2. 인스턴스 길이 5
3. 인스턴스를 무작위로 섞어 80% train / 10% val / 10% test 로 나눈다. 같은 뉴스 재공유는 지우지 않는다.
4. text -> tokens -> emb     save tokens , embs 
   emb tools->  영어 `bert-base-uncased`, 중국어 `bert-base-chinese`.

단어 벡터 파일은 직접 받아서 저장소 폴더(`FakeNewsBench/`)에 둔다 (git 미포함).

- 영어: `glove.840B.300d.zip` — https://nlp.stanford.edu/data/glove.840B.300d.zip
- 중국어: `sgns.merge.word.bz2` — https://github.com/Embedding/Chinese-Word-Vectors 의 Mixed-large / Word

결과 위치 `data/processed/{dataset}/`

| 파일 | 내용 | 사용 모델 |
|---|---|---|
| news_emb.pt | {"title": [N+1,768], "description": [N+1,768]}, row 0 = 패딩 | Rec4Mit, PRISM |
| tokens.pt | title/desc input_ids, attention_mask | HDInt stage1, MINER (BERT fine-tune) |
| category.pt | 뉴스별 카테고리 번호 [N+1]. 제목 TF-IDF → K-means (K = min(300, 뉴스 수 // 100)). CSV에 `category` 컬럼이 있으면(pheme, PHEME 사건 9개) 그 값을 사용 | CAUM, FUM, MINER |
| subcategory.pt | 뉴스별 서브카테고리 번호 [N+1]. 카테고리 안에서 K-means 한 번 더 (K = 카테고리 뉴스 수 // 33, 최소 1) | FUM |
| entity.pt | 뉴스별 엔티티 번호 [N+1, 5]. 제목 NER (영어 `dslim/bert-base-NER`, 중국어 `uer/roberta-base-finetuned-cluener2020-chinese`), 0 = 없음 | CAUM, FUM |
| words.pt | 단어 번호 {"title": [N+1,30], "description": [N+1,50]} + 단어 벡터 표 {"emb": [V+1,300]}. 영어 GloVe, 중국어 Chinese Word Vectors | NRMS, CAUM, FUM, RobustSentiRec |
| sentiment.pt | 뉴스별 제목 감성 점수 [N+1], -1(부정) ~ +1(긍정) | RobustSentiRec |
| pc.pt | Bool [N+1], True = P_c 뉴스 (시간순 앞 20%). 인스턴스·네거티브·추천 후보에서 제외 | 전체 (PRISM 은 분류기 학습에 사용) |
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
- 분할: 인스턴스 무작위 8:1:1 (Rec4Mit 10-fold, HDInt 랜덤, PRISM 유저 단위 → 통일)
- 텍스트: title + description만 (PRISM 본문 text 없음)
-   HDInt용 KeyBERT 키워드 3개, 정치성향 라벨.

| 모델 | 학습 | 원본 대비 |
|---|---|---|
| Rec4Mit | 1단계, 15 epoch | 메타 임베딩은 공통 BERT CLS를 필드별 정규화(`normalize_emb`). 정답이 real인 인스턴스만 예측 손실(`train_real_target_only`). 평가 시 자체 분류기로 fake 후보 제외(`filter_fake`). 적대 손실 1/BCE 가 가끔 폭발하는 건 원본 식(Eq 11) 그대로 |
| HDInt | 2단계: BERT fine-tune 5 → frozen 15 | 키워드 없음 → title 임베딩 ×3 으로 대체. 정치성향 없음 → 전부 중립(1). 둘 다 TODO |
| PRISM | 2단계: IB 분류기 30 → 디퓨전 120 | 분류기는 P_c(시간순 앞 20%) 뉴스로만 학습. 논문은 비율을 밝히지 않음 (2:8 은 우리가 정한 값). 입력 1536-d. 평가 노이즈 시드 고정 |
| NRMS | 1단계, 15 epoch | Keras 원본을 PyTorch로 옮김. 헤드 수는 논문 값 (16×16) |
| MINER | 1단계, 5 epoch (BERT fine-tune) | 저자 공식 코드 없음, 논문 기준으로 구현. 카테고리 임베딩은 GloVe 초기화 대신 처음부터 학습. 워밍업/선형 감쇠 없음. epoch 당 최대 50,000 인스턴스 |
| CAUM | 1단계, 15 epoch | Keras 원본을 PyTorch로 옮김. MIND 엔티티 벡터가 없어 엔티티 임베딩을 학습. 후보마다 유저 벡터를 다시 계산해 full-ranking 평가가 느림 (`eval_every` 5) |
| FUM | 1단계, 15 epoch | Keras 원본을 PyTorch로 옮김. 본문은 description 단어. Fastformer 는 원본 코드의 reshape 순서 대신 논문 식대로 |
| RobustSentiRec | 1단계, 15 epoch | 감성 점수는 원본의 bert_sentiment 방식. 유저 평균 감성은 유저별로 계산 (원본 코드는 배치 전체 평균) |

새로 넣은 다섯 모델은 원본에 없던 패딩 마스크를 어텐션에 넣었다.
