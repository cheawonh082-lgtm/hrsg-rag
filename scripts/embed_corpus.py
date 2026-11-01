"""corpus.json 문단을 BGE-M3로 임베딩해 로컬 벡터 인덱스(index/)로 저장한다.

    python scripts/embed_corpus.py --corpus data/corpus.json --out index/bge-m3

- 문단 분할은 바꾸지 않는다. corpus.json의 문단 하나 = 벡터 하나.
- meta.json에 모델·차원·코퍼스 버전(파일 해시)을 기록한다. 코퍼스가 갱신되면 다시 실행한다.
- --cache를 주면 이미 임베딩한 텍스트는 캐시에서 꺼낸다 (코퍼스 구성 간 공유, 논문 추가 시 새 청크만 계산).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.corpus import FieldMap, corpus_version, load_corpus  # noqa: E402
from rag.embed_cache import EmbeddingCache, text_key  # noqa: E402
from rag.embedder import MODEL_ID, BGEM3Embedder  # noqa: E402
from rag.vector_index import LocalVectorIndex  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus.json")
    ap.add_argument("--out", default="index/bge-m3")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=8192, help="문단 최대 토큰 (BGE-M3 최대 8192)")
    ap.add_argument("--device", default=None, help="cpu | cuda (기본: 자동)")
    ap.add_argument("--cache", help="임베딩 캐시 폴더 (예: index/cache/bge-m3)")
    ap.add_argument("--id-field")
    ap.add_argument("--doc-field")
    ap.add_argument("--text-field")
    ap.add_argument("--page-field")
    args = ap.parse_args(argv)

    paragraphs = load_corpus(args.corpus, FieldMap(args.id_field, args.doc_field, args.text_field, args.page_field))
    version = corpus_version(args.corpus)
    print(f"코퍼스 {len(paragraphs)}개 문단, 문헌 {len({p.doc_id for p in paragraphs})}건 (버전 {version})")

    embedder = BGEM3Embedder(args.model, batch_size=args.batch_size, max_length=args.max_length, device=args.device)
    cache = EmbeddingCache(args.cache) if args.cache else None
    keys = [text_key(args.model, args.max_length, p.text) for p in paragraphs]
    todo = [i for i, k in enumerate(keys) if cache is None or cache.get(k) is None]
    if cache is not None:
        print(f"캐시 {len(cache)}개 중 재사용 {len(paragraphs) - len(todo)}개 · 새로 임베딩 {len(todo)}개")
    started = time.perf_counter()
    fresh: dict[int, np.ndarray] = {}
    step = max(args.batch_size * 8, 1)
    for start in range(0, len(todo), step):
        idx = todo[start : start + step]
        for i, v in zip(idx, embedder.embed([paragraphs[i].text for i in idx])):
            fresh[i] = v
        if cache is not None:  # 중간에 멈춰도 계산한 만큼은 남긴다
            cache.add([keys[i] for i in idx], np.asarray([fresh[i] for i in idx]))
            cache.save()
        print(f"  {min(start + step, len(todo))}/{len(todo)} ({time.perf_counter() - started:.0f}s)", flush=True)
    vectors = np.vstack([fresh[i] if i in fresh else cache.get(k) for i, k in enumerate(keys)])

    index = LocalVectorIndex.build(
        paragraphs,
        vectors,
        meta={"model": args.model, "backend": embedder.backend if fresh else "cache", "corpus_version": version,
              "metric": "cosine", "max_length": args.max_length},
    )
    index.save(args.out)
    print(f"저장: {Path(args.out).resolve()} — {vectors.shape[0]}×{vectors.shape[1]}, {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
