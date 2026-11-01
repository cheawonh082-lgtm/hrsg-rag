"""검색 오프라인 평가 — BM25 기준선 vs 벡터 단독 vs 하이브리드(RRF).

    python scripts/evaluate_search.py --corpus data/corpus.json --index index/bge-m3 \
        --questions data/eval_questions.json [--glossary data/glossary.json] [--unit doc]

순서 (협업안 1절 ④):
1. dev 질문으로만 후보 수(top-N)와 RRF k를 격자 탐색해 설정을 고른다 (recall@8 → MRR → 기본값 순으로 우선).
2. 고른 설정으로 test 질문을 한 번 측정해 최종 성능으로 보고한다. test 결과를 보고 설정을 다시 고치지 않는다.
답이 없는 질문은 recall·MRR에서 빼고, 상위 결과의 유사도 점수만 따로 기록한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.bm25 import BM25Searcher, glossary_expander  # noqa: E402
from rag.corpus import FieldMap, corpus_version, language_groups, load_corpus  # noqa: E402
from rag.embedder import BGEM3Embedder, HashingEmbedder  # noqa: E402
from rag.evaluation import Scores, aggregate, load_questions, score_one  # noqa: E402
from rag.hybrid_search import DEFAULT_RRF_K, DEFAULT_TOP_N, HybridSearcher, SearchHit, bm25_only  # noqa: E402
from rag.vector_index import LocalVectorIndex  # noqa: E402

# 협업안 3절에 보고된 기존 BM25 기준선 (답이 있는 94문항, 상위 8개). 측정값과 나란히 보여 준다.
REPORTED_BASELINE = {
    ("all", "all"): (0.73, 0.63),
    ("all", "en"): (0.88, 0.80),
    ("all", "ko"): (0.57, 0.46),
    ("test", "all"): (0.70, 0.58),
}
ROWS = [("all", "all"), ("all", "en"), ("all", "ko"), ("dev", "all"), ("dev", "ko"), ("test", "all"), ("test", "en"), ("test", "ko")]


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus.json")
    ap.add_argument("--index", default="index/bge-m3")
    ap.add_argument("--questions", nargs="+", default=["data/eval_questions.json"], help="평가 질문 파일 (여러 개 가능)")
    ap.add_argument("--glossary", help="기존 앱의 한국어→영어 용어 목록 JSON (BM25 질의 확장)")
    ap.add_argument("--unit", choices=["doc", "page"], default="doc", help="정답 판정 단위 (기준선과 같게)")
    ap.add_argument("--top-n-grid", default="30,40,50")
    ap.add_argument("--k-grid", default="20,40,60,80,100")
    ap.add_argument("--baseline-per-doc", type=int, default=0, help="기준선 BM25에도 문헌당 제한을 둘 때 (기본 0=없음)")
    ap.add_argument("--lang-balanced", action="store_true", help="벡터 검색을 한국어/영어 문단별로 따로 해 RRF에 넣는다")
    ap.add_argument("--embedder", choices=["bge-m3", "hashing"], default="bge-m3", help="hashing은 동작 확인용")
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--id-field")
    ap.add_argument("--doc-field")
    ap.add_argument("--text-field")
    ap.add_argument("--page-field")
    args = ap.parse_args(argv)

    paragraphs = load_corpus(args.corpus, FieldMap(args.id_field, args.doc_field, args.text_field, args.page_field))
    page_of = {p.paragraph_id: p.page for p in paragraphs}
    index = LocalVectorIndex.load(args.index)
    version = corpus_version(args.corpus)
    if index.meta.get("corpus_version") not in (None, version):
        sys.exit(f"인덱스가 다른 코퍼스 버전({index.meta['corpus_version']})으로 만들어졌습니다. embed_corpus.py를 다시 실행하세요.")
    if args.unit == "page" and all(p is None for p in page_of.values()):
        sys.exit("--unit page를 쓰려면 corpus.json에 페이지 필드가 필요합니다(--page-field).")

    questions = [q for path in args.questions for q in load_questions(path)]
    if len({q.qid for q in questions}) != len(questions):
        sys.exit("질문 ID가 파일 간에 중복됩니다.")
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    dev = [q for q in answerable if q.split == "dev"]
    print(f"질문 {len(questions)}개 (답 있음 {len(answerable)}, 답 없음 {len(unanswerable)}) · dev {len(dev)} · test {sum(q.split == 'test' for q in answerable)}")
    print(f"정답 판정 단위: {args.unit} · 코퍼스 {len(paragraphs)}문단 (버전 {version}) · 인덱스 모델 {index.meta.get('model')}")

    glossary = json.loads(Path(args.glossary).read_text(encoding="utf-8")) if args.glossary else None
    bm25 = BM25Searcher(paragraphs, query_expander=glossary_expander(glossary) if glossary else None)
    embedder = HashingEmbedder(index.vectors.shape[1]) if args.embedder == "hashing" else BGEM3Embedder(index.meta.get("model", "BAAI/bge-m3"))
    hybrid = HybridSearcher(index, embedder, bm25, lang_groups=language_groups(paragraphs) if args.lang_balanced else None)

    # 질문 벡터는 한 번만 만든다 (격자 탐색 중 재사용)
    qvec = dict(zip((q.qid for q in questions), embedder.embed([q.question for q in questions])))

    def run(qs, search) -> tuple[dict, list[dict]]:
        rows, details = [], []
        for q in qs:
            hits: list[SearchHit] = search(q)
            recall, rr = score_one([(h.doc_id, page_of.get(h.paragraph_id)) for h in hits], q, args.unit)
            rows.append((q, recall, rr))
            details.append({"qid": q.qid, "lang": q.lang, "split": q.split, "recall": recall, "rr": rr,
                            "gold": sorted(q.gold_docs), "retrieved": [[h.doc_id, h.paragraph_id] for h in hits]})
        return aggregate(rows), details

    baseline_search = lambda q: bm25_only(bm25, q.question, per_doc=args.baseline_per_doc or None)  # noqa: E731

    # 1) dev 튜닝
    print("\n[dev 튜닝] top-N × RRF k")
    grid = []
    for top_n in ints(args.top_n_grid):
        for k in ints(args.k_grid):
            h = hybrid.with_params(top_n=top_n, rrf_k=k)
            s = run(dev, lambda q, h=h: h.search(q.question, qvec[q.qid]))[0][("dev", "all")]
            grid.append((s.recall, s.mrr, (top_n, k) == (DEFAULT_TOP_N, DEFAULT_RRF_K), top_n, k))
            print(f"  top-N {top_n:>3}  k {k:>3}  recall@8 {s.recall:.3f}  MRR {s.mrr:.3f}")
    best = max(grid)
    top_n, k = best[3], best[4]
    print(f"→ 선택: top-N {top_n}, k {k} (dev recall@8 {best[0]:.3f}, MRR {best[1]:.3f})")
    tuned = hybrid.with_params(top_n=top_n, rrf_k=k)

    # 2) 최종 측정 (test는 여기서 한 번)
    methods = {
        "BM25 (측정)": baseline_search,
        "벡터 단독": lambda q: tuned.search_vector_only(q.question, qvec[q.qid]),
        "하이브리드": lambda q: tuned.search(q.question, qvec[q.qid]),
    }
    results = {name: run(answerable, fn) for name, fn in methods.items()}

    print("\n## 결과 (recall@8 / MRR, 답이 있는 질문)")
    header = "| 구분 | n | 협업안 기준선 | " + " | ".join(methods) + " |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for key in ROWS:
        table0 = results["BM25 (측정)"][0]
        if key not in table0:
            continue
        ref = REPORTED_BASELINE.get(key)
        cells = [f"{results[m][0][key].recall:.2f} / {results[m][0][key].mrr:.2f}" for m in methods]
        print(f"| {key[0]}·{key[1]} | {table0[key].n} | {f'{ref[0]:.2f} / {ref[1]:.2f}' if ref else '–'} | " + " | ".join(cells) + " |")
    for m in methods:
        t: dict[tuple[str, str], Scores] = results[m][0]
        print(f"정답 문헌을 하나도 못 찾은 질문 — {m}: 한국어 {t.get(('all', 'ko'), Scores()).zero_hit}, 영어 {t.get(('all', 'en'), Scores()).zero_hit}")

    # 3) 답이 없는 질문 관찰 (자료 없음 판단 기준 참고용)
    top1 = lambda q: (tuned.search(q.question, qvec[q.qid]) or [None])[0]  # noqa: E731
    answerable_top = [h.vector_score for q in answerable if (h := top1(q)) and h.vector_score is not None]
    observed = []
    for q in unanswerable:
        h = top1(q)
        observed.append({"qid": q.qid, "lang": q.lang, "question": q.question,
                         "top1_cosine": h.vector_score if h else None, "top1_rrf": h.score if h else None})
    if unanswerable:
        print("\n## 답이 없는 질문 — 상위 1위 코사인 유사도")
        if answerable_top:
            print(f"(참고: 답이 있는 질문 평균 {mean(answerable_top):.3f}, 최소 {min(answerable_top):.3f})")
        for o in observed:
            c = f"{o['top1_cosine']:.3f}" if o["top1_cosine"] is not None else "–"
            print(f"- [{o['lang']}] {o['qid']}: {c}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "unit": args.unit, "corpus_version": version, "index_model": index.meta.get("model"),
        "tuned": {"top_n": top_n, "rrf_k": k, "per_doc": tuned.per_doc, "final_k": tuned.final_k},
        "dev_grid": [{"top_n": g[3], "rrf_k": g[4], "recall": g[0], "mrr": g[1]} for g in grid],
        "summary": {m: {f"{s}/{l}": {"n": v.n, "recall@8": v.recall, "mrr": v.mrr, "zero_hit": v.zero_hit}
                        for (s, l), v in results[m][0].items()} for m in methods},
        "per_question": {m: results[m][1] for m in methods},
        "unanswerable": observed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n상세 결과: {out.resolve()}")


if __name__ == "__main__":
    main()
