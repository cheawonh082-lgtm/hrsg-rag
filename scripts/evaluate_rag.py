"""RAG 검색 평가 — 한국어 질문과 영어 질문을 따로 보고한다.

    python scripts/evaluate_rag.py                      # eval/rag_eval_{ko,en}.jsonl → eval/RESULTS.md, results/eval_rag.json

질문마다 검색기(bge·qwen·bm25)를 한 번씩 돌리고, 그 결과를 1~4안이 공유한다(엔진의 compare와 같은 방식).
정답: gold_doc(논문) · gold_ids(문단). 지표는 최종 결과 상위 k개 기준.
- 논문 hit@1 / hit@5, 논문 MRR@5 : 정답 논문의 문단이 몇 위에 처음 나왔는가
- 문단 hit@5 / hit@10           : 정답 문단 자체가 들어왔는가
질문 언어와 정답 문단 언어가 다른 경우(교차언어)와 같은 경우(단일언어)를 나눠서도 본다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.rag_config import RagSettings  # noqa: E402
from rag.rag_engine import METHODS, RETRIEVERS, RagEngine  # noqa: E402

LABELS = {"1": "1안 bge+BM25", "2": "2안 qwen+BM25", "3": "3안 bge+qwen+BM25", "4": "4안 언어별 가중",
          "bge": "bge 단독", "qwen": "qwen 단독", "bm25": "BM25 단독"}
DEPTH = 10  # 문단 hit@10까지 보려고 최종 10개를 받는다


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_rank(ids: list[str], target) -> int | None:
    return next((r for r, x in enumerate(ids, start=1) if target(x)), None)


def score(ranked: list[str], q: dict, doc_of: dict) -> dict:
    d = first_rank(ranked, lambda pid: doc_of[pid] == q["gold_doc"])
    p = first_rank(ranked, lambda pid: pid in q["gold_ids"])
    return {"doc_rank": d, "para_rank": p}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    hit = lambda key, k: mean(1.0 if r[key] and r[key] <= k else 0.0 for r in rows)  # noqa: E731
    return {"n": len(rows), "doc_hit1": hit("doc_rank", 1), "doc_hit5": hit("doc_rank", 5),
            "doc_mrr5": mean(1 / r["doc_rank"] if r["doc_rank"] and r["doc_rank"] <= 5 else 0.0 for r in rows),
            "para_hit5": hit("para_rank", 5), "para_hit10": hit("para_rank", 10)}


def table(summary: dict[str, dict], names: list[str]) -> list[str]:
    lines = ["| 방식 | 논문 hit@1 | 논문 hit@5 | 논문 MRR@5 | 문단 hit@5 | 문단 hit@10 |", "|---|---|---|---|---|---|"]
    best = max(summary[n]["doc_mrr5"] for n in names if n in METHODS)
    for n in names:
        s = summary[n]
        name = f"**{LABELS[n]}**" if n in METHODS and s["doc_mrr5"] == best else LABELS[n]
        lines.append(f"| {name} | {s['doc_hit1']:.2f} | {s['doc_hit5']:.2f} | {s['doc_mrr5']:.2f} | "
                     f"{s['para_hit5']:.2f} | {s['para_hit10']:.2f} |")
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ko", default="eval/rag_eval_ko.jsonl")
    ap.add_argument("--en", default="eval/rag_eval_en.jsonl")
    ap.add_argument("--report", default="eval/RESULTS.md")
    ap.add_argument("--out", default="results/eval_rag.json")
    args = ap.parse_args(argv)

    s = RagSettings.from_env()
    engine = RagEngine(s)
    missing = [m for m, ok in engine.available_methods().items() if not ok]
    if missing:
        sys.exit(f"인덱스가 없어 {missing}안을 평가할 수 없습니다: {engine.missing}")
    doc_of = {p.id: p.doc_id for p in engine.paras}
    para_lang = {p.id: p.lang for p in engine.paras}
    names = [*METHODS, *RETRIEVERS]

    report = {"settings": {"top_k": s.top_k, "rrf_k": s.rrf_k, "weights_ko": s.weights_ko, "weights_en": s.weights_en,
                           "bge_model": s.bge_model, "qwen_model": s.qwen_model},
              "langs": {}}
    md = ["# RAG 검색 평가 결과", "",
          f"코퍼스 문헌 {len(engine.docs)}편 · 문단 {len(engine.paras)}개. 검색기마다 top_k={s.top_k}, RRF k={s.rrf_k}. "
          f"모델: `{s.bge_model}`, `{s.qwen_model}`. 4안 가중치: ko {s.weights_ko}, en {s.weights_en}.",
          "", "정답은 질문마다 문단 1개(와 그 논문)이다. 논문 지표는 정답 논문의 아무 문단이나 상위에 오면 맞은 것으로 본다. "
          "평가 질문은 코퍼스를 보고 작성했으므로 실제 사용자 질문보다 쉬울 수 있다.", ""]
    started = time.perf_counter()
    for lang, path in (("ko", Path(args.ko)), ("en", Path(args.en))):
        qs = load(path)
        rows: dict[str, list[dict]] = {n: [] for n in names}
        detail = []
        for q in qs:
            hits, _ = engine.retrieve(q["query"], RETRIEVERS, s.top_k)
            q["cross"] = para_lang[q["gold_ids"][0]] != lang
            per = {}
            for m in METHODS:
                ranked = [r["id"] for r in engine.search(q["query"], m, top_n=DEPTH, _hits=hits)["results"]]
                per[m] = score(ranked, q, doc_of) | {"top": ranked[:5]}
            for r in RETRIEVERS:
                per[r] = score([h.id for h in hits[r]][:DEPTH], q, doc_of)
            for n in names:
                rows[n].append(per[n] | {"cross": q["cross"]})
            detail.append({"qid": q["qid"], "query": q["query"], "gold_ids": q["gold_ids"], "cross_lingual": q["cross"],
                           "results": per})
            print(f"  {q['qid']} " + " ".join(f"{m}:{per[m]['doc_rank'] or '-'}" for m in METHODS), flush=True)
        summary = {n: summarize(rows[n]) for n in names}
        split = {kind: {n: summarize([r for r in rows[n] if r["cross"] == (kind == "cross")]) for n in names}
                 for kind in ("cross", "same")}
        report["langs"][lang] = {"n": len(qs), "summary": summary, "by_gold_lang": split, "per_question": detail}

        title = "한국어 질문" if lang == "ko" else "영어 질문"
        n_cross = sum(q["cross"] for q in qs)
        md += [f"## {title} ({len(qs)}개)", "",
               f"정답 문단이 {'영어' if lang == 'ko' else '한국어'}인 교차언어 질문 {n_cross}개, 같은 언어 {len(qs) - n_cross}개.", "",
               *table(summary, names), ""]
        for kind, label in (("cross", "교차언어"), ("same", "같은 언어")):
            if split[kind]["1"]:
                md += [f"### {title} — {label} ({split[kind]['1']['n']}개)", "", *table(split[kind], names), ""]
        misses = [d["qid"] for d in detail if not d["results"]["4"]["doc_rank"] or d["results"]["4"]["doc_rank"] > 5]
        md += [f"4안 기준 정답 논문을 상위 5개에서 못 찾은 질문: {', '.join(misses) or '없음'}", ""]

    md += ["## 읽는 법", "",
           "- 굵게 표시한 행이 해당 언어에서 논문 MRR@5가 가장 높은 방식(1~4안 중)이다.",
           "- `bge/qwen/BM25 단독`은 비교용이며 API의 method로는 제공하지 않는다.",
           "- 질문이 언어별 40개 안팎이라 0.05 안쪽 차이는 우연일 수 있다."]
    Path(args.report).write_text("\n".join(md) + "\n", encoding="utf-8")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{(time.perf_counter() - started) / 60:.1f}분 · 보고서 {Path(args.report).resolve()} · 상세 {out.resolve()}")


if __name__ == "__main__":
    main()
