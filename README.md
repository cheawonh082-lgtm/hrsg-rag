# HRSG Explorer — RAG 검색 벡터화·평가

협업안(261021)의 범위: 다국어 임베딩(BGE-M3) + 벡터 인덱스, BM25와의 하이브리드(RRF), 코퍼스 임베딩·평가 스크립트, Cloudflare Workers AI·Vectorize 연동 시험.
UI·프롬프트·계산 로직·청크 분할은 건드리지 않는다.

## RAG 검색 API (`rag_corpus/R???.json`, 41편 · 3,095문단)

```bash
.venv/Scripts/python -m pip install -r requirements-api.txt

# 1) 병합 + 인덱스: rag_corpus/R???.json → data/rag/corpus_all.json → index/rag/bge-m3, index/rag/qwen3-embedding-0.6b
#    CPU 기준 모델당 2~3시간. 배치마다 캐시에 저장되므로 끊겨도 다시 실행하면 이어서 계산한다.
.venv/Scripts/python scripts/build_rag_index.py --src rag_corpus

# 2) 서버
.venv/Scripts/python -m uvicorn api.app:app --port 8100

# 3) 호출
curl -X POST localhost:8100/search -H "Content-Type: application/json" \
     -d '{"query": "HRSG 핀치 포인트를 낮추면 어떻게 되나?", "method": "4", "top_n": 5}'
```

| 검색기 (각 top_k=50) | 내용 |
|---|---|
| `bge` | BAAI/bge-m3 dense, 정규화 후 코사인. 입력 `"[section] text"` |
| `qwen` | Qwen/Qwen3-Embedding (기본 0.6B, `RAG_QWEN_MODEL`). 질의에만 `Instruct: …\nQuery:` 프롬프트, 문서는 그대로 |
| `bm25` | section + text + terms. 한국어는 kiwipiepy 형태소(내용어만), 영어는 소문자화. `IF97`·`CO2`·`HP/IP/LP`·`kJ/kg`·`IAPWS-IF97`·`°C`는 한 토큰(+부분)으로 보호 |

| 방식 (`method`, 최종 top_n=5) | 검색기 → 결합 |
|---|---|
| `1` | bge + bm25 → RRF (k=60) |
| `2` | qwen + bm25 → RRF |
| `3` | bge + qwen + bm25 → RRF |
| `4` | 3안 검색기 → 질문 언어별 가중 RRF. 한글/(한글+라틴) ≥ 0.3이면 ko. 기본 가중치 ko `bge=1,qwen=1,bm25=0`, en도 `bge=1,qwen=1,bm25=0` (`RAG_WEIGHTS_KO/EN`). 평가셋 v1 dev에서 가중치 91개 조합을 비교해 두 언어 모두 1:1:0으로 정했다 |

엔드포인트: `GET /health` · `GET /methods` · `POST /search` · `POST /search/compare`(1~4안 한 번에) · `GET /paragraphs/{id}` · `GET /documents/{doc_id}`.
결과 문단마다 `retrievers`에 검색기별 순위·점수가 있어 어느 검색기가 끌어올렸는지 볼 수 있다. 인덱스가 없는 방식은 503과 이유를 돌려준다.
한국어 질문은 모든 방식에서 한↔영 용어 사전(`rag/glossary.py`)의 영어 용어를 덧붙인 질의로 검색한다(`RAG_EXPAND_KO`, 응답의 `search_query`). 코퍼스 문단의 95%가 영어라 한국어 질문 대부분이 교차언어 검색이 되기 때문이다. 4안 한국어 가중치(BM25 끔)와 함께 평가셋으로 정했다 — `eval/RESULTS.md`.

## 구성

| 파일 | 역할 |
|---|---|
| `rag/corpus.py` | `corpus.json` 로더 (필드 이름 자동 감지, `--id-field` 등으로 지정 가능) · 코퍼스 버전 해시 |
| `rag/embedder.py` | `BAAI/bge-m3` 밀집 임베딩 (1,024차원, L2 정규화 → 내적 = 코사인) |
| `rag/vector_index.py` | 로컬 벡터 인덱스 (`index/<이름>/embeddings.npy` + `meta.json`) |
| `rag/bm25.py` | BM25 + 한국어 용어 목록 확장 — **임시 구현, 기존 앱 BM25로 교체 필요** |
| `rag/hybrid_search.py` | 벡터 top-N + BM25 top-N → RRF(k) → 문헌당 최대 2문단 → 최종 8개 |
| `rag/evaluation.py` | recall@8 · MRR (문헌 단위 기본, 페이지 단위 선택) |
| `rag/cloudflare_vectorize.py` | Workers AI 임베딩 · 로컬↔Workers AI 일치 확인 · Vectorize 생성/업서트/조회 |
| `scripts/embed_corpus.py` | 코퍼스 임베딩 → 로컬 인덱스 저장 |
| `scripts/evaluate_search.py` | dev 튜닝 → test 1회 측정, BM25 기준선과 비교 표 |
| `scripts/cloudflare_sync.py` | Cloudflare 배포 (parity / upload [--model qwen] / deploy / query) |
| `scripts/smoke_crosslingual.py` | 공식 데이터 전, 참고문헌 PDF 일부로 교차 언어 검색 점검 (로컬 전용) |
| `scripts/ingest_pdfs.py` | 참고문헌 PDF를 로컬 실험 코퍼스(`data/local_smoke/`)에 증분 추가·임베딩 (PyMuPDF, 중복·스캔본 건너뜀) |
| `rag/md_corpus.py` | `rag_corpus/`의 EN/KO 병렬 md → 청크 (frontmatter 분리, `##` 절 단위, 700단어 넘는 절은 `###` 단위, EN/KO 같은 청크 ID) · 스펙 검증 |
| `rag/embed_cache.py` | 임베딩 캐시 — 같은 텍스트는 다시 임베딩하지 않음 (구성 간 공유, 논문 추가 시 새 청크만) |
| `scripts/build_corpus_md.py` | md 코퍼스 → `data/corpus_md/{mixed,ko,en_strict}.json` + 평가 질의 정답 존재 점검 |
| `rag/md_eval.py` | md 코퍼스 평가 — jsonl 질의 로더, 모드별 정답, hit@1/3/8·MRR·정답 절·질의 언어 비율 |
| `scripts/evaluate_md.py` | md 코퍼스 평가 실행 (모드 3개 × BM25/벡터/하이브리드/언어균형/pair 중복 제거) |
| `data/eval_questions*.json` | **사용 중지** — 이전 PDF 코퍼스(저작권 문제로 삭제)용 질문. 정답 문헌 대부분이 새 코퍼스에 없음 (U01→P10, U07→P03만 대응) |

## 설치

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # GPU면 생략
.venv/Scripts/python -m pip install -r requirements-embed.txt
```
첫 임베딩 때 Hugging Face에서 `BAAI/bge-m3`(약 2.3GB)를 내려받는다.

## 실행 순서

`data/`에 `corpus.json`, `eval_questions.json`(선택: `glossary.json` — 기존 앱의 한국어→영어 용어 목록)을 둔다.

```bash
# 1) 코퍼스 임베딩 (코퍼스가 갱신되면 다시 실행 — meta.json의 corpus_version으로 검증)
.venv/Scripts/python scripts/embed_corpus.py --corpus data/corpus.json --out index/bge-m3

# 2) 평가: dev로 top-N(30/40/50) × RRF k(20~100) 튜닝 → 선택한 설정으로 test 1회
.venv/Scripts/python scripts/evaluate_search.py --corpus data/corpus.json --index index/bge-m3 \
    --questions data/eval_questions.json --glossary data/glossary.json --unit doc

# 3) Cloudflare 배포 (배포판 앱용 검색 Worker) — 아래 "Cloudflare 배포" 참고
.venv/Scripts/python scripts/cloudflare_sync.py parity
.venv/Scripts/python scripts/cloudflare_sync.py upload          # parity 실패 시 --reembed
.venv/Scripts/python scripts/cloudflare_sync.py deploy
.venv/Scripts/python scripts/cloudflare_sync.py query "감온기 분사수 제어 방식"

# 테스트
.venv/Scripts/python -m pytest -q tests
```

결과는 표준 출력의 표와 `results/eval.json`(질문별 검색 결과·순위 포함)에 남는다.

### md 코퍼스 (`rag_corpus/`, EN/KO 병렬 정리본)

논문을 추가할 때는 `rag_corpus/docs/{en,ko}/`(교차언어 평가용이면 `docs_xling_strict/en/`도)와 `eval/`에 파일을 넣고 아래를 다시 실행한다.
논문 ID는 코드에 없고 폴더를 훑는다. 캐시 덕분에 기존 청크는 다시 임베딩하지 않는다.

```bash
# 1) 적재·검증 → data/corpus_md/{mixed,ko,en_strict}.json (스펙 오류가 있으면 멈춤, 질의 정답 존재 점검)
.venv/Scripts/python scripts/build_corpus_md.py --root rag_corpus --out data/corpus_md

# 2) 임베딩 (구성별 인덱스, 캐시 공유)
for v in mixed ko en_strict; do
  .venv/Scripts/python scripts/embed_corpus.py --corpus data/corpus_md/$v.json --out index/md/$v --cache index/cache/bge-m3
done
```

| 구성 | 문서 | 평가 모드 (rag_corpus/README '권장 측정 방식') |
|---|---|---|
| `mixed` | `docs/en` + `docs/ko` | 혼합·언어 균형 — 한국어 문서만 상위에 오르는지 |
| `ko` | `docs/ko` | 단일언어 — 한글 질의 → 한국어 문서 |
| `en_strict` | `docs_xling_strict/en` | **교차언어(핵심)** — 한글 전용 질의 → 한글 0자 영어 문서 |

```bash
# 3) 평가: 모드마다 dev로 하이브리드 top-N × RRF k 선택 → 전체 측정 (test 행이 최종값)
.venv/Scripts/python scripts/evaluate_md.py --queries rag_corpus/eval_queries_all.jsonl --out results/eval_md.json
```

- 정답: `en_strict` = 영어 문서, `ko` = 한국어 문서, `mixed` = pair 단위(EN·KO 어느 쪽이든). 정답 절은 `gold_section`.
- split: 질의에 `split`이 없으면 qid 해시로 고정 배정(약 절반 dev) — 논문을 추가해도 기존 질의 배정은 그대로.
- 질의 임베딩도 같은 캐시를 쓴다.

## 임베딩에서 지킨 것

- **질문·문단을 같은 방식으로 임베딩**: BGE-M3 밀집 검색은 질문 앞에 지시문(instruction)을 붙이지 않는다(bge-*-v1.5와 다름). 질문과 문단 모두 원문 그대로 넣는다.
- **정규화**: 모든 벡터를 L2 정규화해 내적이 곧 코사인 유사도 — Vectorize `cosine` 인덱스와 점수가 맞는다.
- **풀링**: FlagEmbedding(`dense_vecs`)과 sentence-transformers(모델 설정의 CLS 풀링) 어느 쪽이든 같은 밀집 벡터를 쓴다. 어떤 백엔드로 만들었는지 `meta.json`의 `backend`에 기록.
- **최대 길이**: 코퍼스 임베딩은 기본 8,192토큰(잘림 없음). 문단이 길면 CPU 시간이 늘어나므로 `--max-length`로 줄일 수 있지만, 질문·문단 모두 같은 값을 쓴다.
- **환경 간 일치**: Workers AI(`@cf/baai/bge-m3`)로 만든 질문 벡터를 로컬 코퍼스 벡터와 섞기 전에 `cloudflare_sync.py parity`로 같은 문장의 코사인(기준 0.99)을 확인한다. Workers AI가 긴 입력을 다르게 자를 수 있어 긴 문단도 표본에 포함한다. 일치하지 않으면 코퍼스를 Workers AI로 다시 임베딩한다.
- **규격명 검색**: BM25 토크나이저는 `IAPWS-IF97`을 전체와 부분(`iapws`, `if97`)으로 모두 색인해 "IF97"만으로도 찾는다.

## md 코퍼스 평가 결과 (2026-10-27, 논문 10편 = 문서 20개·214청크, 질의 140개)

한글 전용 질의 60개 (`results/eval_md.json`):

| 모드 | 방법 | hit@1 | hit@8 | MRR | 정답 절 MRR |
|---|---|---|---|---|---|
| 교차언어 (`en_strict`) | BM25 | 0.00 | 0.00 | 0.00 | 0.00 |
| | 벡터 = 하이브리드 | 0.63 | 0.98 | 0.75 | 0.33 |
| 단일언어 (`ko`) | BM25 | 0.87 | 1.00 | 0.93 | 0.20 |
| | 벡터 | 0.83 | 0.98 | 0.88 | 0.34 |
| | 하이브리드 | 0.85 | 1.00 | 0.92 | 0.37 |

- 교차언어 검색은 벡터(BGE-M3)만 가능하다. 언어를 넘으면 hit@1이 0.83 → 0.63으로 떨어진다.
- 1위를 놓친 22개의 공통점: 한글로 쓴 숫자·음차 고유명사("삼백오십 메가와트", "튀르키예 복합화력")와 주제가 겹치는 엑서지 논문(P02·P07·P09) 간 혼동.
- `mixed`에서는 한국어 쌍둥이 문서 때문에 BM25가 가장 높다(pair MRR 0.97) — 임베딩 성능은 `en_strict`로만 본다.
  언어 균형은 질의 언어 청크 비율을 70% → 49%로 맞추지만 MRR이 소폭 낮다. pair 중복 제거는 순위를 바꾸지 않고 같은 논문의 EN/KO 중복을 없앤다.
- 문서가 10~20개라 hit@8은 포화(≈1.00) — hit@1·MRR·정답 절로 비교한다. 후보 수 50이 코퍼스 대부분이라 dev 튜닝도 아직 의미가 작다.
- 정답 절 지표는 같은 수치가 1·5·6절에 반복돼 엄격하게 나온다.
- 긴 표 행 단위 분할(README_1 권장)은 하지 않았다: P02·P06 6절은 이미 `###`로 7·4청크로 나뉘었고, 6절 정답 질의가 9개뿐이라 비교가 어렵다. 논문 추가 후 다시 본다.

## 이전 PDF 코퍼스 실험 메모 (원문은 저작권 문제로 삭제, 참고용)

- 영어 문헌만 있을 때는 영어 용어 없는 한국어 질문 10개 중 8개가 영어 정답 문단을 찾았다.
- 한국어 문헌(59문단)이 섞이자 **한국어 질문이 한국어 문헌을 상위로 끌어올려 영어 정답 문헌이 밀리는 언어 편향**이 보였고, 영어 질문은 한국어 정답 문헌을 거의 못 찾았다.
- `--lang-balanced`(언어별 벡터 검색 후 RRF)는 재현율을 높이고(특히 영어→한국어 문헌), MRR은 낮춘다. 기본은 꺼짐 — 공식 평가셋에서 비교 후 결정.
- 현재 BM25는 임시 구현이라 하이브리드가 벡터 단독보다 낮게 나온다. 기존 앱 BM25(용어 목록 확장)로 바꾼 뒤 다시 비교해야 한다.

## 평가 기준

- 정답 판정은 기본 **문헌 단위**(정답 문헌의 아무 문단이면 정답). `--unit page`는 (문헌, 근거 페이지) 일치. 기준선과 같은 단위를 써야 공정하므로 결과 파일에 단위를 기록한다.
- 답이 없는 질문 6개는 recall·MRR에서 빼고, 하이브리드 1위의 코사인 유사도를 답이 있는 질문의 분포와 함께 기록한다('자료 없음' 판단 기준 참고).
- test 결과를 본 뒤 설정을 다시 고치지 않는다.

## 받아야 할 것 (현재 미보유)

1. 추가 논문 30여 편의 EN/KO md(+ `docs_xling_strict/en`)와 평가 질의 — 같은 명령으로 다시 적재·임베딩·평가
2. 기존 앱의 BM25 코드와 한국어 용어 목록 — `KeywordSearcher` 프로토콜(`search(query, top_n)`)에 맞춘 어댑터로 바꿔 끼운다. 기준선(전체 0.73 / 0.63)을 먼저 재현한 뒤 비교한다.

## 보안

`.gitignore`: `*.pdf`, `.env`, `index/`, `results/`, `data/local_*/`, `data/corpus_md/`. API 토큰은 환경변수로만 읽는다.

## Cloudflare 배포 (배포판 앱의 논문 검색)

배포판 앱(Render 무료 플랜)은 임베딩 모델을 올릴 수 없으므로, 검색을 Cloudflare Worker(`cloudflare/worker`)에서 한다.

- **질문 임베딩:** Workers AI `@cf/baai/bge-m3`와 `@cf/qwen/qwen3-embedding-0.6b`(질문에 `RAG_QWEN_INSTRUCTION`). 한국어 질문은 `rag/glossary.py`의 영어 용어를 덧붙인다(`src/query.js`가 같은 규칙).
- **검색:** Vectorize 인덱스 `hrsg-bge-m3`(1,024차원)와 `hrsg-qwen3-06b-512`(Qwen3 MRL 512차원, cosine). 무료 플랜 저장 한도가 계정당 500만 차원이라 qwen은 앞 512차원만 쓴다(평가셋 v1 dev에서 1,024차원과 같거나 나았다). 문단 본문과 서지 정보는 두 인덱스 모두 메타데이터에 있다(`paragraph_metadata`, 10 KiB 이하).
- **응답:** `POST /search`는 이 저장소 RAG API와 같은 형식이라, 앱은 `RAG_API_URL`·`RAG_API_KEY`만 바꾸면 된다. 모든 요청에 `Authorization: Bearer <RAG_API_KEY>`가 필요하다.
- **검색 방식:** 로컬 4안과 같은 bge:qwen:BM25 = 1:1:0. 검색기마다 상위 50개를 가중 RRF(k=60)로 합친다(`weightedRrf`, Python `weighted_rrf`와 같은 동점 처리). 가중치·instruction은 `worker-data`가 `rag_config`에서 `src/search_config.json`으로 내보내며, BM25 가중치가 0이 아니면 배포를 멈춘다.

**자격 증명** (`.env`, Git 제외)
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`: 권한은 Account의 Workers Scripts Edit · Workers AI Read · Vectorize Edit · Account Settings Read
- `RAG_API_KEY`: `deploy`가 없으면 만들어 `.env`에 저장한다. 같은 값을 앱(Render)의 `RAG_API_KEY`에 넣는다.

**순서**
1. `parity`: 로컬 벡터와 Workers AI 벡터가 같은지(코사인 0.99 이상) 확인한다. 다르면 `upload --reembed`로 올린다.
2. `upload`(bge)와 `upload --model qwen`(로컬 qwen 벡터 앞 512차원. Workers AI 질문 벡터와 코사인 1.0 확인함): 반영은 비동기라 1~2분 뒤 `query`로 확인한다.
3. `deploy`
4. 서지 정보를 고치면 `upload`와 `upload --model qwen`을 다시 한다.

**Worker 테스트:** `cloudflare_sync.py worker-data` 후 `node --test cloudflare/worker/test/`
