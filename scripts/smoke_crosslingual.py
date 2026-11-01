"""교차 언어 임베딩 점검 (임시 — 공식 corpus.json·평가셋을 받기 전 동작 확인용).

참고문헌 PDF 일부에서 문단을 뽑아 로컬 전용 코퍼스(data/local_smoke/, Git 제외)를 만들고,
영어 용어를 넣지 않은 한국어 질문으로 영어 문단을 찾는지 BM25(용어 목록 없음) · 벡터 · 하이브리드를 비교한다.

판정: 상위 8개 문단 중 하나라도 목표 영어 용어(정규식)를 포함하면 '적중' — 정답 문헌 라벨이 없어 쓰는 대리 지표.
⚠ 이 코퍼스의 문단 분할은 임시(페이지별 단어 창)이며 공식 corpus.json과 다르다. 수치는 기준선과 비교하지 않는다.

    python scripts/smoke_crosslingual.py --pdf-dir "C:/Users/chaewon/Downloads/HRSG_RAG_참고문헌"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.bm25 import BM25Searcher  # noqa: E402
from rag.corpus import Paragraph  # noqa: E402
from rag.embedder import BGEM3Embedder  # noqa: E402
from rag.hybrid_search import HybridSearcher, bm25_only  # noqa: E402
from rag.vector_index import LocalVectorIndex  # noqa: E402

DEFAULT_FILES = [
    "IAPWS-IF97-Rev2007.pdf", "IAPWS-TGD3-10.pdf", "IAPWS-TGD7-16.pdf", "Dooley-Lister-FAC-2018.pdf",
    "GE-GER-3574G.pdf", "Ansaldo_HRSG.pdf", "IEAGHG_2012-08.pdf", "JPCS2019_pinch.pdf",
    "Gartia-2007-IJTS-natural-circulation-parallel-channels.pdf", "energies-08-01101.pdf",
    "JAKO200819463917983.pdf", "Kim2002_KSMEB.pdf",
]

# (질문, 목표 용어 정규식, 언어) — 한국어 질문에는 영어 용어·약어를 넣지 않는다
PROBES = [
    ("감온기 분사수로 과열 증기 온도를 어떻게 조절하나요?", r"attemperat|desuperheat|spray water", "ko"),
    ("배관 안의 빠른 물 흐름 때문에 생기는 부식은 왜 발생하나요?", r"flow[- ]accelerated corrosion", "ko"),
    ("증발기 출구 가스 온도와 포화 온도의 최소 차이", r"pinch", "ko"),
    ("드럼 보일러에서 펌프 없이 물이 순환하는 원리", r"natural circulation", "ko"),
    ("절탄기에서 급수를 예열하는 역할", r"economi[sz]er", "ko"),
    ("산업용 증기 물성 계산식 국제 표준", r"industrial formulation|IF97", "ko"),
    ("이산화탄소 포집 설비를 붙인 복합화력의 효율 저하", r"capture", "ko"),
    ("덕트 버너로 추가 연소할 때의 영향", r"supplementary firing|duct burner|duct firing", "ko"),
    ("보일러 수 화학 관리와 산소 처리", r"oxygenated treatment|water chemistry|cycle chemistry", "ko"),
    ("가스터빈 배기 압력 손실이 출력에 주는 영향", r"back[- ]?pressure|pressure drop|pressure loss", "ko"),
    ("attemperator spray water control of superheater temperature", r"attemperat|desuperheat", "en"),
    ("flow-accelerated corrosion in HRSG low pressure evaporators", r"flow[- ]accelerated corrosion", "en"),
]


def extract(pdf: Path, words: int, max_chunks: int) -> list[Paragraph]:
    from pypdf import PdfReader

    out: list[Paragraph] = []
    for page_no, page in enumerate(PdfReader(str(pdf)).pages, start=1):
        tokens = re.sub(r"\s+", " ", page.extract_text() or "").split(" ")
        for start in range(0, len(tokens), words):
            chunk = " ".join(tokens[start : start + words]).strip()
            if len(chunk) >= 200:
                out.append(Paragraph(f"{pdf.stem[:40]}#p{page_no}-{start // words}", pdf.stem, chunk, page_no))
            if len(out) >= max_chunks:
                return out
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--out", default="data/local_smoke")
    ap.add_argument("--words", type=int, default=150, help="임시 분할: 페이지별 단어 창 크기")
    ap.add_argument("--max-chunks", type=int, default=150, help="문헌당 최대 문단 수")
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args(argv)
    out = Path(args.out)

    index_dir = out / "index"
    if (index_dir / "meta.json").exists():
        index = LocalVectorIndex.load(index_dir)
        paragraphs = [Paragraph(**p) for p in json.loads((out / "corpus.json").read_text(encoding="utf-8"))]
        print(f"기존 로컬 코퍼스 사용: {len(paragraphs)}문단")
    else:
        paragraphs = []
        for name in args.files:
            pdf = Path(args.pdf_dir) / name
            if pdf.exists():
                try:
                    chunks = extract(pdf, args.words, args.max_chunks)
                except Exception as e:  # 암호화·손상 PDF는 건너뛴다
                    print(f"  {name}: 건너뜀 ({type(e).__name__})")
                    continue
                paragraphs += chunks
                print(f"  {name}: {len(chunks)}문단")
        out.mkdir(parents=True, exist_ok=True)
        (out / "corpus.json").write_text(json.dumps([p.__dict__ for p in paragraphs], ensure_ascii=False), encoding="utf-8")
        print(f"임베딩 {len(paragraphs)}문단 (CPU) …", flush=True)
        t = time.perf_counter()
        vectors = BGEM3Embedder(max_length=args.max_length, batch_size=8).embed([p.text for p in paragraphs])
        print(f"  {time.perf_counter() - t:.0f}s")
        index = LocalVectorIndex.build(paragraphs, vectors, {"model": "BAAI/bge-m3", "note": "local smoke corpus"})
        index.save(index_dir)

    text_of = {p.paragraph_id: p.text for p in paragraphs}
    embedder = BGEM3Embedder(max_length=args.max_length)
    bm25 = BM25Searcher(paragraphs)  # 용어 목록 없이 — 기존 앱의 한국어 한계를 그대로 보기 위해
    hybrid = HybridSearcher(index, embedder, bm25, top_n=40, rrf_k=60)
    qvecs = embedder.embed([p[0] for p in PROBES])

    def hit(hits, pattern) -> int | None:
        return next((r for r, h in enumerate(hits, start=1) if re.search(pattern, text_of[h.paragraph_id], re.I)), None)

    print("\n| 언어 | 질문 | BM25 | 벡터 | 하이브리드 | 하이브리드 1위 문헌 (코사인) |")
    print("|---|---|---|---|---|---|")
    totals = {"ko": [0, 0, 0, 0], "en": [0, 0, 0, 0]}
    for (question, pattern, lang), qv in zip(PROBES, qvecs):
        b = hit(bm25_only(bm25, question), pattern)
        v = hit(hybrid.search_vector_only(question, qv), pattern)
        hh = hybrid.search(question, qv)
        h = hit(hh, pattern)
        t = totals[lang]
        t[0] += 1
        t[1] += b is not None
        t[2] += v is not None
        t[3] += h is not None
        fmt = lambda r: f"{r}위" if r else "✗"  # noqa: E731
        top = f"{hh[0].doc_id[:28]} ({hh[0].vector_score:.2f})" if hh and hh[0].vector_score is not None else (hh[0].doc_id[:28] if hh else "–")
        print(f"| {lang} | {question[:34]} | {fmt(b)} | {fmt(v)} | {fmt(h)} | {top} |")
    for lang, (n, b, v, h) in totals.items():
        print(f"\n{lang}: 상위 8개 적중 — BM25 {b}/{n}, 벡터 {v}/{n}, 하이브리드 {h}/{n}")


if __name__ == "__main__":
    main()
