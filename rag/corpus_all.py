"""RAG 코퍼스 (corpus_all.json) — 문헌별 R???.json을 합치고, 검색에 쓰는 입력 문자열을 만든다.

corpus_all.json: {"documents": [...], "paragraphs": [...]}
- documents[]: doc_id, title, authors, year, venue, doi, url, license, source_type, lang (+ 매핑 파일의 source_file)
- paragraphs[]: id, doc_id, section, page_start, page_end, text, terms, content_type, source_type

문헌 언어(documents[].lang)와 문단 언어는 다를 수 있다 (한국어 논문의 영어 초록·캡션 등).
그래서 문단마다 본문의 한글 비율로 판별한 lang도 함께 둔다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.lang import detect_lang

CONTENT_TYPES = frozenset({"text", "table", "equation", "caption"})
SOURCE_TYPES = frozenset({"verbatim", "paraphrase"})
PARA_FIELDS = ("id", "doc_id", "section", "page_start", "page_end", "text", "terms", "content_type", "source_type")
DOC_FIELDS = ("doc_id", "title", "authors", "year", "venue", "doi", "url", "license", "source_type", "lang")


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    lang: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    license: str = ""
    source_type: str = ""
    source_file: str = ""


@dataclass(frozen=True)
class Para:
    id: str
    doc_id: str
    section: str
    page_start: int | None
    page_end: int | None
    text: str
    terms: tuple[str, ...]
    content_type: str
    source_type: str
    lang: str  # 문단 본문으로 판별한 언어

    @property
    def embed_input(self) -> str:
        """임베딩 입력: "[section] text" (bge·qwen 문서 쪽 공통, instruction 없음)"""
        return f"[{self.section}] {self.text}" if self.section else self.text

    @property
    def bm25_input(self) -> str:
        """BM25 검색 대상: section + text + terms"""
        return " ".join([self.section, self.text, " ".join(self.terms)])


def merge_sources(src_dir: str | Path, mapping: str | Path | None = None) -> tuple[dict, list[str]]:
    """R???.json (각각 {"document", "paragraphs", "report"}) → corpus_all dict. (데이터, 경고) — 오류면 ValueError."""
    src = Path(src_dir)
    files = sorted(p for p in src.glob("*.json") if "mapping" not in p.name and p.name != "corpus_all.json")
    if not files:
        raise FileNotFoundError(f"{src}에 문헌 JSON이 없습니다")
    meta = {}
    if mapping is None:
        mapping = next(iter(sorted(src.glob("*mapping*.json"))), None)
    if mapping:
        meta = {m["doc_id"]: m for m in json.loads(Path(mapping).read_text(encoding="utf-8"))}

    errors, warnings = [], []
    documents, paragraphs, seen = [], [], set()
    for f in files:
        raw = json.loads(f.read_text(encoding="utf-8"))
        doc, paras = raw.get("document"), raw.get("paragraphs")
        if not isinstance(doc, dict) or not isinstance(paras, list):
            errors.append(f"{f.name}: document/paragraphs가 없습니다")
            continue
        missing = [k for k in DOC_FIELDS if k not in doc]
        if missing:
            errors.append(f"{f.name}: document 필드 누락 {missing}")
        m = meta.get(doc.get("doc_id"), {})
        if meta and not m:
            warnings.append(f"{f.name}: 매핑 파일에 {doc.get('doc_id')}가 없습니다")
        if m and m.get("paragraph_count") not in (None, len(paras)):
            warnings.append(f"{f.name}: 문단 수 {len(paras)} ≠ 매핑 {m['paragraph_count']}")
        documents.append({**doc, "source_file": m.get("source_file", "")})
        for p in paras:
            missing = [k for k in PARA_FIELDS if k not in p]
            if missing:
                errors.append(f"{f.name} {p.get('id')}: 문단 필드 누락 {missing}")
                continue
            if p["id"] in seen:
                errors.append(f"{f.name}: 문단 ID {p['id']} 중복")
            seen.add(p["id"])
            if p["doc_id"] != doc.get("doc_id"):
                errors.append(f"{f.name} {p['id']}: doc_id {p['doc_id']}가 문헌과 다릅니다")
            if p["content_type"] not in CONTENT_TYPES or p["source_type"] not in SOURCE_TYPES:
                errors.append(f"{f.name} {p['id']}: content_type/source_type 값 오류")
            if not str(p["text"]).strip():
                warnings.append(f"{f.name} {p['id']}: 본문이 비어 있습니다")
            paragraphs.append(p)
    if errors:
        raise ValueError("코퍼스 오류:\n- " + "\n- ".join(errors[:50]))
    return {"documents": documents, "paragraphs": paragraphs}, warnings


def load_corpus_all(path: str | Path) -> tuple[dict[str, Doc], list[Para]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = {
        d["doc_id"]: Doc(
            doc_id=d["doc_id"], title=d.get("title") or "", lang=d.get("lang") or "", authors=tuple(d.get("authors") or ()),
            year=d.get("year"), venue=d.get("venue") or "", doi=d.get("doi") or "", url=d.get("url") or "",
            license=d.get("license") or "", source_type=d.get("source_type") or "", source_file=d.get("source_file") or "",
        )
        for d in raw["documents"]
    }
    paras = [
        Para(
            id=p["id"], doc_id=p["doc_id"], section=p.get("section") or "", page_start=p.get("page_start"),
            page_end=p.get("page_end"), text=p["text"], terms=tuple(p.get("terms") or ()),
            content_type=p.get("content_type", "text"), source_type=p.get("source_type", ""), lang=detect_lang(p["text"]),
        )
        for p in raw["paragraphs"]
    ]
    unknown = {p.doc_id for p in paras} - docs.keys()
    if unknown:
        raise ValueError(f"documents에 없는 doc_id: {sorted(unknown)}")
    return docs, paras
