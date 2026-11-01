"""corpus.json 로더. 청크 분할은 바꾸지 않고 파일의 문단을 그대로 쓴다.

실제 corpus.json의 필드 이름을 아직 받지 못해, 흔한 이름을 차례로 찾고(자동 감지)
CLI에서 --id-field 등으로 직접 지정할 수도 있게 했다. 맞는 필드가 없으면 사용 가능한 키를 보여 주고 멈춘다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ID_FIELDS = ("paragraph_id", "chunk_id", "para_id", "pid", "id")
DOC_FIELDS = ("doc_id", "ref_id", "reference_id", "source_id", "paper_id", "doc")
TEXT_FIELDS = ("text", "content", "paragraph", "body", "chunk")
PAGE_FIELDS = ("page", "page_no", "page_number")


@dataclass(frozen=True)
class Paragraph:
    paragraph_id: str
    doc_id: str
    text: str
    page: int | None = None


@dataclass(frozen=True)
class FieldMap:
    id: str | None = None
    doc: str | None = None
    text: str | None = None
    page: str | None = None


def _records(raw) -> list[dict]:
    """[{...}] 또는 {"paragraphs": [...]} / {"chunks": [...]} 형태를 모두 받는다."""
    if isinstance(raw, list):
        return raw
    for key in ("paragraphs", "chunks", "corpus", "items", "data"):
        if isinstance(raw.get(key), list):
            return raw[key]
    raise ValueError(f"corpus.json에서 문단 목록을 찾지 못했습니다. 최상위 키: {sorted(raw)}")


def _pick(record: dict, explicit: str | None, candidates: tuple[str, ...], what: str, required: bool = True) -> str | None:
    if explicit:
        if explicit not in record:
            raise KeyError(f"{what} 필드 '{explicit}'가 없습니다. 사용 가능한 키: {sorted(record)}")
        return explicit
    found = next((c for c in candidates if c in record), None)
    if found is None and required:
        raise KeyError(f"{what} 필드를 찾지 못했습니다(후보 {candidates}). 사용 가능한 키: {sorted(record)} — CLI 옵션으로 지정하세요.")
    return found


def load_corpus(path: str | Path, fields: FieldMap = FieldMap()) -> list[Paragraph]:
    records = _records(json.loads(Path(path).read_text(encoding="utf-8")))
    if not records:
        raise ValueError("corpus.json에 문단이 없습니다.")
    first = records[0]
    id_f = _pick(first, fields.id, ID_FIELDS, "문단 ID")
    doc_f = _pick(first, fields.doc, DOC_FIELDS, "문헌 ID")
    text_f = _pick(first, fields.text, TEXT_FIELDS, "본문")
    page_f = _pick(first, fields.page, PAGE_FIELDS, "페이지", required=False)

    paragraphs = [
        Paragraph(
            paragraph_id=str(r[id_f]),
            doc_id=str(r[doc_f]),
            text=str(r[text_f]),
            page=int(r[page_f]) if page_f and r.get(page_f) is not None else None,
        )
        for r in records
    ]
    ids = [p.paragraph_id for p in paragraphs]
    if len(set(ids)) != len(ids):
        raise ValueError("문단 ID가 중복됩니다. --id-field를 확인하세요.")
    return paragraphs


def corpus_version(path: str | Path) -> str:
    """코퍼스 파일 해시 — 임베딩이 어느 버전의 코퍼스로 만들어졌는지 기록·검증한다."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def language_groups(paragraphs: list[Paragraph], ko_ratio: float = 0.2) -> dict[str, list[str]]:
    """본문의 한글 비율로 문단을 한국어/영어로 나눈다 (언어 균형 검색용)."""
    groups: dict[str, list[str]] = {"ko": [], "en": []}
    for p in paragraphs:
        hangul = sum("가" <= ch <= "힣" for ch in p.text)
        groups["ko" if hangul > ko_ratio * max(len(p.text), 1) else "en"].append(p.paragraph_id)
    return groups
