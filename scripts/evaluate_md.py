"""md 코퍼스 오프라인 평가 (3단계) — 모드(en_strict / ko / mixed) × 방법(BM25 / 벡터 / 하이브리드 …).

    python scripts/evaluate_md.py --queries rag_corpus/eval_queries_all.jsonl

입력: build_corpus_md.py가 만든 data/corpus_md/{mode}.json, embed_corpus.py가 만든 index/md/{mode}.
순서 (협업안 1절 ④와 같음):
1. 모드마다 dev 질의로만 하이브리드 후보 수(top-N)와 RRF k를 고른다.
2. 고른 설정으로 전체 질의를 측정한다. test 행이 최종 보고값이며, test를 보고 설정을 다시 고치지 않는다.
질의 split은 파일의 split 필드, 없으면 qid 해시로 고정 배정한다(rag/md_eval.default_split).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.bm25 import BM25Searcher  # noqa: E402
from rag.corpus import corpus_version, load_corpus  # noqa: E402
from rag.embed_cache import EmbeddingCache, text_key  # noqa: E402
from rag.embedder import BGEM3Embedder, HashingEmbedder  # noqa: E402
from rag.hybrid_search import DEFAULT_RRF_K, DEFAULT_TOP_N, HybridSearcher, bm25_only  # noqa: E402
from rag.md_eval import MODES, ChunkInfo, MdQuery, load_md_queries, score_query, summarize  # noqa: E402
from rag.vector_index import LocalVectorIndex  # noqa: E402

ROW_ORDER = ["한글전용", "기본-ko", "기본-en", "all", "dev", "test"]


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def embed_queries(queries: list[MdQuery], args) -> dict[str, np.ndarray]:
    if args.embedder == "hashing":
        emb = HashingEmbedder(1024)
        return dict(zip((q.qid for q in queries), emb.embed([q.query for q in queries])))
    emb = BGEM3Embedder(args.model, max_length=args.max_length)
    cache = EmbeddingCache(args.cache) if args.cache else None
    keys = [text_key(args.model, args.max_length, q.query) for q in queries]
    todo = [i for i, k in enumerate(keys) if cache is None or cache.get(k) is None]
    print(f"질의 임베딩: {len(queries)}개 중 새로 {len(todo)}개" + (" (나머지 캐시)" if cache else ""))
    fresh = dict(zip(todo, emb.embed([queries[i].query for i in todo]))) if todo else {}
    if cache is not None and fresh:
        cache.add([keys[i] for i in fresh], np.asarray(list(fresh.values())))
        cache.save()
    return {q.qid: fresh[i] if i in fresh else cache.get(keys[i]) for i, q in enumerate(queries)}


def fmt(x: float) -> str:
    return "–" if x != x else f"{x:.2f}"


def run_mode(mode: str, queries: list[MdQuery], qvec: dict, args) -> dict:
    corpus_path = Path(args.corpus_dir) / f"{mode}.json"
    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    paragraphs = load_corpus(corpus_path)
    index = LocalVectorIndex.load(Path(args.index_dir) / mode)
    if index.meta.get("corpus_version") not in (None, corpus_version(corpus_path)):
        sys.exit(f"[{mode}] 인덱스와 코퍼스 버전이 다릅니다. embed_corpus.py를 다시 실행하세요.")
    info = {c["paragraph_id"]: ChunkInfo(c["doc_id"], c["pair_id"], c["section"], c["lang"]) for c in raw["paragraphs"]}
    pair_of = {c.doc_id: c.pair_id for c in info.values()}

    bm25 = BM25Searcher(paragraphs)
    base = HybridSearcher(index, None, bm25)
    dev = [q for q in queries if q.split == "dev"]

    def evaluate(qs, search) -> tuple[dict, list]:
        rows, detail = [], []
        for q in qs:
            ranked = [h.paragraph_id for h in search(q)]
            s = score_query(ranked, q, mode, info)
            rows.append((q, s))
            detail.append({"qid": q.qid, "rank": s.rank, "section_rank": s.section_rank, "top": ranked})
        return summarize(rows), detail

    # 1) dev 튜닝 (하이브리드)
    grid = []
    for top_n in ints(args.top_n_grid):
        for k in ints(args.k_grid):
            h = base.with_params(top_n=top_n, rrf_k=k)
            s = evaluate(dev, lambda q, h=h: h.search(q.query, qvec[q.qid]))[0]["all"]
            grid.append((s.hit(8), s.mrr, s.section_mrr, (top_n, k) == (DEFAULT_TOP_N, DEFAULT_RRF_K), top_n, k))
    best = max(grid)
    tuned = base.with_params(top_n=best[4], rrf_k=best[5])
    print(f"\n### [{mode}] 문서 {raw['meta']['n_docs']} · 청크 {len(paragraphs)} · dev 튜닝 → top-N {best[4]}, k {best[5]} "
          f"(dev hit@8 {best[0]:.2f}, MRR {best[1]:.2f})")

    # 2) 측정
    methods = {
        "BM25": lambda q: bm25_only(bm25, q.query, per_doc=tuned.per_doc),
        "벡터": lambda q: tuned.search_vector_only(q.query, qvec[q.qid]),
        "하이브리드": lambda q: tuned.search(q.query, qvec[q.qid]),
    }
    if mode == "mixed":
        groups = {"ko": [c for c, i in info.items() if i.lang == "ko"], "en": [c for c, i in info.items() if i.lang == "en"]}
        balanced = tuned.with_params(lang_groups=groups)
        dedupe = tuned.with_params(doc_group=pair_of)
        methods["벡터·언어균형"] = lambda q: balanced.search_vector_only(q.query, qvec[q.qid])
        methods["하이브리드·pair중복제거"] = lambda q: dedupe.search(q.query, qvec[q.qid])
    results = {name: evaluate(queries, fn) for name, fn in methods.items()}

    t0 = results["BM25"][0]
    rows = [r for r in ROW_ORDER if r in t0]
    print("\n문서 hit@1 / hit@8 / MRR" + (" (pair 단위)" if mode == "mixed" else ""))
    print("| 질의 | n | " + " | ".join(methods) + " |")
    print("|---|---|" + "---|" * len(methods))
    for r in rows:
        cells = [f"{fmt(results[m][0][r].hit(1))} / {fmt(results[m][0][r].hit(8))} / {fmt(results[m][0][r].mrr)}" for m in methods]
        print(f"| {r} | {t0[r].n} | " + " | ".join(cells) + " |")
    print("\n정답 절 hit@8 / 절 MRR" + (" · 질의 언어 청크 비율" if mode == "mixed" else ""))
    print("| 질의 | " + " | ".join(methods) + " |")
    print("|---|" + "---|" * len(methods))
    for r in rows:
        cells = []
        for m in methods:
            s = results[m][0][r]
            cells.append(f"{fmt(s.section_hit8)} / {fmt(s.section_mrr)}" + (f" · {s.same_lang:.0%}" if mode == "mixed" else ""))
        print(f"| {r} | " + " | ".join(cells) + " |")
    types = sorted(k for k in t0 if k.startswith("한글전용·"))
    if types:
        print("\n한글 전용 질의 유형별 (벡터 / 하이브리드 / BM25, hit@8 · MRR)")
        for t in types:
            v, h, b = (results[m][0][t] for m in ("벡터", "하이브리드", "BM25"))
            print(f"- {t.split('·')[1]} (n={v.n}): {fmt(v.hit(8))}·{fmt(v.mrr)} / {fmt(h.hit(8))}·{fmt(h.mrr)} / {fmt(b.hit(8))}·{fmt(b.mrr)}")

    return {
        "corpus": raw["meta"], "tuned": {"top_n": best[4], "rrf_k": best[5], "per_doc": tuned.per_doc, "final_k": tuned.final_k},
        "dev_grid": [{"top_n": g[4], "rrf_k": g[5], "hit@8": g[0], "mrr": g[1], "section_mrr": g[2]} for g in grid],
        "summary": {m: {k: v.as_dict() for k, v in res[0].items()} for m, res in results.items()},
        "per_query": {m: res[1] for m, res in results.items()},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", nargs="+", default=["rag_corpus/eval_queries_all.jsonl"])
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--corpus-dir", default="data/corpus_md")
    ap.add_argument("--index-dir", default="index/md")
    ap.add_argument("--cache", default="index/cache/bge-m3", help="질의 임베딩 캐시 (빈 값이면 끔)")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--max-length", type=int, default=8192, help="코퍼스 임베딩과 같은 값")
    ap.add_argument("--top-n-grid", default="30,40,50")
    ap.add_argument("--k-grid", default="20,40,60,80,100")
    ap.add_argument("--embedder", choices=["bge-m3", "hashing"], default="bge-m3", help="hashing은 동작 확인용")
    ap.add_argument("--out", default="results/eval_md.json")
    args = ap.parse_args(argv)

    queries = load_md_queries(args.queries)
    n_dev = sum(q.split == "dev" for q in queries)
    print(f"질의 {len(queries)}개 (dev {n_dev}, test {len(queries) - n_dev}) — "
          + ", ".join(f"{g} {sum(q.group == g for q in queries)}" for g in ROW_ORDER[:3]))
    qvec = embed_queries(queries, args)
    out = {"queries": args.queries, "modes": {}}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        out["modes"][mode] = run_mode(mode, queries, qvec, args)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n상세 결과: {path.resolve()}")


if __name__ == "__main__":
    main()
