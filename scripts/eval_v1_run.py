"""HRSG 검색 평가셋 v1 (hrsg_eval_set_v1)에 우리 검색 방식 1~4안을 돌려 결과 JSON을 만든다.

    python scripts/eval_v1_run.py                       # 임베딩(없으면 계산·캐시) → .cache/eval/<label>.json
    node App/tools/report_by_lang.mjs BM25=.cache/eval/bm25.json ...   # 채점·보고는 평가셋 스크립트로

평가셋 규칙(hrsg_eval_set_v1/CLAUDE.md)을 따른다.
- eval_questions.json, corpus.json은 읽기만 한다. 채점 지표는 만들지 않고 결과 JSON(ranked)만 쓴다.
- 상위 8개, 한 참고문헌당 최대 2개: retrieve.js의 cap()처럼 점수 순으로 훑으며 3번째부터 건너뛴다.
- 임베딩은 로컬 모델로만 계산하고 <평가셋>/.cache/embeddings/에 저장한다. 키가 (모델, 최대 길이, 텍스트)
  해시라 모델이나 corpus가 바뀌면 바뀐 passage만 다시 계산한다. 질문 벡터는 실행할 때마다 새로 계산한다.

검색 설정은 RAG API와 같다 (rag_config.RagSettings, 각 검색기 top_k=50, RRF k=60, 한국어 질문 용어 확장,
4안 언어별 가중치). passage에 section이 없어 임베딩·BM25 입력은 text만 쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.bm25 import BM25Searcher  # noqa: E402
from rag.corpus import Paragraph  # noqa: E402
from rag.embed_cache import EmbeddingCache, text_key  # noqa: E402
from rag.embedder import BGEM3Embedder, Qwen3Embedder  # noqa: E402
from rag.glossary import expand_query  # noqa: E402
from rag.lang import detect_lang  # noqa: E402
from rag.rag_config import RagSettings, model_slug  # noqa: E402
from rag.rag_engine import METHODS, weighted_rrf  # noqa: E402
from rag.tokenizer import tokenize  # noqa: E402

K, CAP = 8, 2
LABELS = {"1": "m1-bge-bm25", "2": "m2-qwen-bm25", "3": "m3-bge-qwen-bm25", "4": "m4-lang-weighted"}


def embed_passages(name: str, embedder, model_id: str, texts: list[str], s: RagSettings, cache_dir: Path, step: int) -> np.ndarray:
    cache = EmbeddingCache(cache_dir / model_slug(model_id))
    keys = [text_key(model_id, s.max_length, t) for t in texts]
    todo = sorted((i for i, k in enumerate(keys) if cache.get(k) is None), key=lambda i: len(texts[i]))
    print(f"[{name}] {model_id}: passage {len(texts)} · 캐시 재사용 {len(texts) - len(todo)} · 새로 {len(todo)}", flush=True)
    started = time.perf_counter()
    done_chars, total_chars = 0, sum(len(texts[i]) for i in todo) or 1
    for start in range(0, len(todo), step):
        idx = todo[start: start + step]
        cache.add([keys[i] for i in idx], embedder.embed([texts[i] for i in idx]))
        cache.save()  # 배치마다 저장 → 중단해도 이어서
        done_chars += sum(len(texts[i]) for i in idx)
        elapsed = time.perf_counter() - started
        print(f"  {min(start + step, len(todo))}/{len(todo)}  {elapsed / 60:.1f}분 경과, "
              f"남은 시간 약 {elapsed / done_chars * (total_chars - done_chars) / 60:.0f}분", flush=True)
    return np.vstack([cache.get(k) for k in keys])


def cap(fused: list[str], ref_of: dict[str, str]) -> list[str]:
    """점수 순으로 훑으며 한 참고문헌의 3번째 passage부터 건너뛰고 K개까지 (retrieve.js와 같은 방식)"""
    per: dict[str, int] = {}
    out = []
    for pid in fused:
        n = per.get(ref_of[pid], 0)
        if n >= CAP:
            continue
        per[ref_of[pid]] = n + 1
        out.append(pid)
        if len(out) >= K:
            break
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="hrsg_eval_set_v1", help="평가셋 폴더 (App/ 가 있는 곳)")
    ap.add_argument("--methods", default="1,2,3,4")
    ap.add_argument("--batch", type=int, default=64, help="캐시 저장 단위(passage 수)")
    ap.add_argument("--label", help="결과 파일 이름(방식을 하나만 줄 때). 기본은 LABELS — 설정을 바꿔 다시 돌릴 때 이전 결과를 덮지 않게")
    ap.add_argument("--qwen-dims", type=int, help="Qwen 벡터 앞 N차원만 쓰고 다시 정규화 (MRL, 배포 Worker는 512)")
    args = ap.parse_args(argv)
    if args.label and "," in args.methods:
        ap.error("--label은 --methods에 방식을 하나만 줄 때 쓴다")
    root = Path(args.root).resolve()
    s = RagSettings.from_env()

    corpus = json.loads((root / "App/web-server/data/corpus.json").read_text(encoding="utf-8"))
    questions = json.loads((root / "App/tools/eval_questions.json").read_text(encoding="utf-8"))
    passages = corpus["passages"]
    ids = [f"p{i}" for i in range(len(passages))]
    ref_of = {pid: p["ref"] for pid, p in zip(ids, passages)}
    texts = [p["text"] for p in passages]

    cache_dir = root / ".cache/embeddings"
    vectors = {
        "bge": embed_passages("bge", bge := BGEM3Embedder(s.bge_model, max_length=s.max_length, device=s.device),
                              s.bge_model, texts, s, cache_dir, args.batch),
        "qwen": embed_passages("qwen", qwen := Qwen3Embedder(s.qwen_model, s.qwen_instruction, max_length=s.max_length,
                                                              device=s.device), s.qwen_model, texts, s, cache_dir, args.batch),
    }
    embedders = {"bge": bge, "qwen": qwen}
    mrl = (lambda v: _truncate(v, args.qwen_dims)) if args.qwen_dims else (lambda v: v)  # noqa: E731
    if args.qwen_dims:
        vectors["qwen"] = mrl(vectors["qwen"])
    bm25 = BM25Searcher([Paragraph(pid, ref_of[pid], t) for pid, t in zip(ids, texts)], tokenizer=tokenize)

    runs: dict[str, list[dict]] = {m: [] for m in args.methods.split(",")}
    for q in questions:
        lang = detect_lang(q["question"], s.ko_ratio)
        query = expand_query(q["question"]) if s.expand_ko and lang == "ko" else q["question"]
        qvec = {"bge": embedders["bge"].embed_queries([query])[0], "qwen": mrl(embedders["qwen"].embed_queries([query]))[0]}
        hits = {r: [ids[i] for i in np.argsort(-(vectors[r] @ qvec[r]), kind="stable")[:s.top_k]] for r in ("bge", "qwen")}
        hits["bm25"] = [h.paragraph_id for h in bm25.search(query, s.top_k)]
        for m in runs:
            spec = METHODS[m]
            weights = (s.weights_ko if lang == "ko" else s.weights_en) if spec["weighted"] else None
            fused = [pid for pid, _ in weighted_rrf({r: hits[r] for r in spec["retrievers"]}, s.rrf_k, weights)]
            ranked = cap(fused, ref_of)
            runs[m].append({"id": q["id"], "ranked": [
                {"rank": i, "ref": passages[int(pid[1:])]["ref"], "page": passages[int(pid[1:])].get("page"),
                 "kind": passages[int(pid[1:])]["kind"]} for i, pid in enumerate(ranked, start=1)]})
        print(f"  {q['id']} ({lang})", flush=True)

    out_dir = root / ".cache/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for m, rows in runs.items():
        label = args.label or LABELS[m]
        path = out_dir / f"{label}.json"
        setup = {"weights_ko": s.weights_ko, "weights_en": s.weights_en, "expand_ko": s.expand_ko, "qwen_dims": args.qwen_dims}
        path.write_text(json.dumps({"k": K, "method": m, "label": METHODS[m]["label"], "settings": setup, "questions": rows},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{label} → {path}")


def _truncate(v: np.ndarray, dims: int) -> np.ndarray:
    """Qwen3-Embedding MRL — 앞 dims 차원만 남기고 L2 정규화 (cloudflare_sync.py mrl, Worker truncateNormalize와 같다)"""
    v = np.asarray(v, dtype=np.float32)[..., :dims]
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


if __name__ == "__main__":
    main()
