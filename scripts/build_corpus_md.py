"""rag_corpus/의 EN/KO md 문서 → 평가 모드별 corpus.json (1단계: 적재·검증).

    python scripts/build_corpus_md.py --root rag_corpus --out data/corpus_md

만드는 파일 (README '권장 측정 방식'):
- mixed.json      docs/en + docs/ko          혼합·언어 균형
- ko.json         docs/ko                    단일언어
- en_strict.json  docs_xling_strict/en       교차언어 (한글 0자 영어본)

논문이 추가되면 같은 명령으로 다시 만든다. 스펙 오류(frontmatter 누락, 절 번호, EN/KO 구조 불일치)가 있으면 멈추고,
평가 질의의 정답 문서·정답 절이 코퍼스에 없으면 함께 알려 준다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.md_corpus import DEFAULT_SPLIT_WORDS, VARIANTS, build  # noqa: E402


def check_queries(path: Path, corpora: dict[str, dict]) -> list[str]:
    """질의의 gold_doc_ids가 mixed 코퍼스에, gold_section 청크가 그 문서에 있는지."""
    mixed = corpora.get("mixed")
    if mixed is None or not path.exists():
        return []
    sections = {(c["doc_id"], c["section"]) for c in mixed["paragraphs"]}
    docs = {d["doc_id"] for d in mixed["docs"]}
    problems, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        if q["qid"] in seen:
            problems.append(f"{q['qid']}: qid 중복")
        seen.add(q["qid"])
        for doc in q.get("gold_doc_ids", []):
            if doc not in docs:
                problems.append(f"{q['qid']}: 정답 문서 {doc}가 코퍼스에 없습니다")
            elif q.get("gold_section") and (doc, str(q["gold_section"])) not in sections:
                problems.append(f"{q['qid']}: {doc}에 {q['gold_section']}절이 없습니다")
    return problems


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="rag_corpus")
    ap.add_argument("--out", default="data/corpus_md")
    ap.add_argument("--variants", default=",".join(VARIANTS), help=f"쉼표로 구분 ({', '.join(VARIANTS)})")
    ap.add_argument("--split-words", type=int, default=DEFAULT_SPLIT_WORDS, help="영어본 절이 이 단어 수를 넘으면 ### 단위로 나눔")
    ap.add_argument("--queries", default="eval_queries_all.jsonl", help="--root 기준 평가 질의 파일 (정답 존재 점검)")
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    corpora = {}
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        try:
            corpora[variant] = build(root, variant, args.split_words)
        except ValueError as e:
            sys.exit(f"[{variant}] {e}")

    out.mkdir(parents=True, exist_ok=True)
    print("| 구성 | 문서 | 청크 | 언어 | 나눈 절 | 청크 단어 수 (최소/중앙/최대) |")
    print("|---|---|---|---|---|---|")
    for variant, data in corpora.items():
        chunks = data["paragraphs"]
        n = sorted(c["n_words"] for c in chunks)
        langs = Counter(c["lang"] for c in chunks)
        split = sorted({f"{c['pair_id']} {c['section']}절" for c in chunks if c["sub"]})
        print(f"| {variant} | {data['meta']['n_docs']} | {len(chunks)} | {dict(langs)} | {', '.join(split) or '-'} "
              f"| {n[0]} / {n[len(n) // 2]} / {n[-1]} |")
        (out / f"{variant}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        for w in data["meta"]["warnings"]:
            print(f"  경고 [{variant}] {w}")

    problems = check_queries(root / args.queries, corpora)
    print(f"\n평가 질의 점검 ({args.queries}): " + ("이상 없음" if not problems else f"{len(problems)}건"))
    for p in problems:
        print(f"- {p}")
    print(f"저장: {out.resolve()}")


if __name__ == "__main__":
    main()
