"""참고문헌 PDF를 로컬 실험 코퍼스(data/local_smoke/)에 추가하고 새 문단만 임베딩한다 (증분).

    python scripts/ingest_pdfs.py --pdf-dir "C:/Users/chaewon/Downloads/HRSG_RAG_참고문헌"

- 이미 들어 있는 문헌(doc_id = 파일 이름)은 건너뛰고, 내용이 같은 중복 PDF(MD5)도 한 번만 넣는다.
- 텍스트는 PyMuPDF로 추출한다 (pypdf가 못 읽는 한국어 글꼴 KSCms-UHC 대응). 글자가 없는 스캔 PDF는 OCR이 필요해 건너뛴다.
- 분할은 smoke 코퍼스와 같은 임시 방식(페이지별 150단어 창)이다. 공식 corpus.json을 받으면 그것을 쓴다.
- 문헌을 몇 편 처리할 때마다 저장해(체크포인트) 중간에 멈춰도 이어서 실행할 수 있다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.corpus import Paragraph  # noqa: E402
from rag.embedder import BGEM3Embedder  # noqa: E402
from rag.vector_index import LocalVectorIndex  # noqa: E402


def extract(pdf: Path, words: int) -> list[Paragraph]:
    import pymupdf

    out: list[Paragraph] = []
    with pymupdf.open(pdf) as doc:
        for page_no, page in enumerate(doc, start=1):
            tokens = re.sub(r"\s+", " ", page.get_text()).strip().split(" ")
            for start in range(0, len(tokens), words):
                chunk = " ".join(tokens[start : start + words]).strip()
                if len(chunk) >= 200:
                    out.append(Paragraph(f"{pdf.stem[:40]}#p{page_no}-{start // words}", pdf.stem, chunk, page_no))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--out", default="data/local_smoke")
    ap.add_argument("--words", type=int, default=150)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--checkpoint-docs", type=int, default=3, help="이 편수마다 저장")
    args = ap.parse_args(argv)

    out = Path(args.out)
    corpus_path, index_dir = out / "corpus.json", out / "index"
    paragraphs = [Paragraph(**p) for p in json.loads(corpus_path.read_text(encoding="utf-8"))] if corpus_path.exists() else []
    index = LocalVectorIndex.load(index_dir) if (index_dir / "meta.json").exists() else None
    if index is not None and index.paragraph_ids != [p.paragraph_id for p in paragraphs]:
        sys.exit("corpus.json과 인덱스의 문단 순서가 다릅니다.")
    vectors = index.vectors if index is not None else np.zeros((0, 1024), dtype=np.float32)

    done = {p.doc_id for p in paragraphs}
    seen_hash: dict[str, str] = {}
    todo: list[Path] = []
    for pdf in sorted(Path(args.pdf_dir).glob("*.pdf")):
        digest = hashlib.md5(pdf.read_bytes()).hexdigest()
        if digest in seen_hash:
            print(f"중복 건너뜀: {pdf.name} = {seen_hash[digest]}")
            continue
        seen_hash[digest] = pdf.name
        if pdf.stem not in done:
            todo.append(pdf)
    print(f"기존 {len(done)}편 {len(paragraphs)}문단 · 추가 대상 {len(todo)}편")

    embedder = BGEM3Embedder(max_length=args.max_length, batch_size=8)
    started = time.perf_counter()
    pending = 0
    skipped: list[str] = []
    for i, pdf in enumerate(todo, start=1):
        try:
            chunks = extract(pdf, args.words)
        except Exception as e:
            skipped.append(f"{pdf.name} ({type(e).__name__})")
            continue
        if not chunks:
            skipped.append(f"{pdf.name} (텍스트 없음 — 스캔본, OCR 필요)")
            continue
        t = time.perf_counter()
        new_vecs = embedder.embed([c.text for c in chunks])
        paragraphs += chunks
        vectors = np.vstack([vectors, new_vecs])
        pending += 1
        print(f"[{i}/{len(todo)}] {pdf.name}: {len(chunks)}문단, {time.perf_counter() - t:.0f}s (누적 {time.perf_counter() - started:.0f}s)", flush=True)
        if pending >= args.checkpoint_docs or i == len(todo):
            _save(paragraphs, vectors, corpus_path, index_dir)
            pending = 0
    _save(paragraphs, vectors, corpus_path, index_dir)
    print(f"\n완료: {len({p.doc_id for p in paragraphs})}편 {len(paragraphs)}문단")
    for s in skipped:
        print(f"건너뜀: {s}")


def _save(paragraphs: list[Paragraph], vectors: np.ndarray, corpus_path: Path, index_dir: Path) -> None:
    corpus_path.write_text(json.dumps([p.__dict__ for p in paragraphs], ensure_ascii=False), encoding="utf-8")
    LocalVectorIndex.build(paragraphs, vectors, {"model": "BAAI/bge-m3", "note": "local smoke corpus (임시 분할)"}).save(index_dir)


if __name__ == "__main__":
    main()
