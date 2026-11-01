"""RAG 인덱스 구축 — 문헌 JSON 병합 → corpus_all.json → bge / qwen 임베딩 인덱스.

    python scripts/build_rag_index.py --src rag_corpus                 # 병합 + bge + qwen
    python scripts/build_rag_index.py --src rag_corpus --models bge    # 한 모델만
    RAG_QWEN_MODEL=Qwen/Qwen3-Embedding-4B python scripts/build_rag_index.py --models qwen

- 임베딩 입력: "[section] text" (문서 쪽에는 instruction 없음). BM25는 API 시작 때 메모리에 만든다.
- 캐시(index/cache/<모델>)에 배치마다 저장 — 중간에 멈춰도 다시 실행하면 이어서 계산하고,
  문헌이 추가돼도 새 문단만 임베딩한다.
- 길이가 비슷한 문단끼리 배치를 묶어 CPU 패딩 낭비를 줄인다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.corpus import corpus_version  # noqa: E402
from rag.corpus_all import load_corpus_all, merge_sources  # noqa: E402
from rag.embed_cache import EmbeddingCache, text_key  # noqa: E402
from rag.embedder import BGEM3Embedder, Qwen3Embedder  # noqa: E402
from rag.rag_config import RagSettings, model_slug  # noqa: E402
from rag.vector_index import LocalVectorIndex  # noqa: E402


def embed_all(name: str, embedder, model_id: str, paras, s: RagSettings, out_dir: Path, step: int, extra_meta: dict) -> None:
    texts = [p.embed_input for p in paras]
    cache = EmbeddingCache(s.cache_dir / model_slug(model_id))
    keys = [text_key(model_id, s.max_length, t) for t in texts]
    todo = sorted((i for i, k in enumerate(keys) if cache.get(k) is None), key=lambda i: len(texts[i]))
    print(f"[{name}] {model_id}: 문단 {len(texts)} · 캐시 재사용 {len(texts) - len(todo)} · 새로 {len(todo)}", flush=True)
    started = time.perf_counter()
    done_chars, total_chars = 0, sum(len(texts[i]) for i in todo) or 1
    for start in range(0, len(todo), step):
        idx = todo[start: start + step]
        cache.add([keys[i] for i in idx], embedder.embed([texts[i] for i in idx]))
        cache.save()  # 배치마다 저장 → 중단해도 이어서
        done_chars += sum(len(texts[i]) for i in idx)
        elapsed = time.perf_counter() - started
        eta = elapsed / done_chars * (total_chars - done_chars)
        print(f"  {min(start + step, len(todo))}/{len(todo)}  {elapsed / 60:.1f}분 경과, 남은 시간 약 {eta / 60:.0f}분", flush=True)
    vectors = np.vstack([cache.get(k) for k in keys])
    index = LocalVectorIndex(vectors, [p.id for p in paras], [p.doc_id for p in paras], meta={
        "model": model_id, "metric": "cosine", "max_length": s.max_length, "input": "[section] text",
        "corpus_version": corpus_version(s.corpus), **extra_meta})
    index.save(out_dir)
    print(f"[{name}] 저장 {out_dir} — {vectors.shape[0]}×{vectors.shape[1]}, {(time.perf_counter() - started) / 60:.1f}분", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="문헌 JSON 폴더 (R???.json + *mapping*.json). 주면 corpus_all.json을 새로 만든다")
    ap.add_argument("--models", default="bge,qwen", help="bge,qwen 중 쉼표로")
    ap.add_argument("--batch", type=int, default=64, help="캐시 저장 단위(문단 수)")
    args = ap.parse_args(argv)
    s = RagSettings.from_env()

    if args.src:
        data, warnings = merge_sources(args.src)
        s.corpus.parent.mkdir(parents=True, exist_ok=True)
        s.corpus.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"병합: 문헌 {len(data['documents'])} · 문단 {len(data['paragraphs'])} → {s.corpus}")
        for w in warnings:
            print(f"  경고: {w}")
    docs, paras = load_corpus_all(s.corpus)
    by_lang = {lang: sum(p.lang == lang for p in paras) for lang in ("ko", "en")}
    print(f"코퍼스 {s.corpus.name}: 문헌 {len(docs)} · 문단 {len(paras)} (문단 언어 {by_lang})")

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if name == "bge":
            emb = BGEM3Embedder(s.bge_model, max_length=s.max_length, device=s.device)
            embed_all(name, emb, s.bge_model, paras, s, s.bge_index, args.batch, {})
        elif name == "qwen":
            emb = Qwen3Embedder(s.qwen_model, s.qwen_instruction, max_length=s.max_length, device=s.device)
            embed_all(name, emb, s.qwen_model, paras, s, s.qwen_index, args.batch,
                      {"query_instruction": s.qwen_instruction})
        else:
            sys.exit(f"알 수 없는 모델: {name}")


if __name__ == "__main__":
    main()
