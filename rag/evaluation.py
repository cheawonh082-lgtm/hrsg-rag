"""오프라인 검색 평가: recall@8, MRR.

정답 판정 단위(--unit)
- doc  (기본): 결과 문단의 문헌 ID가 정답 문헌에 있으면 정답. 협업안 3절 계산 예시와 같은 문헌 단위.
- page : (문헌 ID, 페이지)가 근거 페이지와 일치해야 정답. corpus.json에 페이지 필드가 있어야 한다.
기준선과 같은 단위로 비교해야 공정하므로 결과 파일에 단위를 함께 기록한다.

recall@8 = (상위 8개에 들어온 정답 수) / (그 질문의 전체 정답 수)
MRR      = 평균(1 / 첫 정답 순위), 상위 8개 안에 없으면 0
답이 없는 질문(정답 없음)은 두 지표에서 빼고 따로 관찰한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable, Literal, Sequence

Unit = Literal["doc", "page"]

QUESTION_FIELDS = ("question", "query", "q", "text")
ID_FIELDS = ("id", "qid", "question_id")
LANG_FIELDS = ("lang", "language", "locale")
SPLIT_FIELDS = ("split", "set", "subset")
GOLD_DOC_FIELDS = ("gold_docs", "answer_docs", "relevant_docs", "doc_ids", "gold", "answers", "references")
EVIDENCE_FIELDS = ("evidence", "evidence_pages", "gold_pages", "pages")


@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    question: str
    lang: str  # "ko" | "en"
    split: str  # "dev" | "test"
    gold_docs: frozenset[str]
    gold_pages: frozenset[tuple[str, int]] = frozenset()

    @property
    def answerable(self) -> bool:
        return bool(self.gold_docs)


@dataclass
class Scores:
    recalls: list[float] = field(default_factory=list)
    rrs: list[float] = field(default_factory=list)
    zero_hit: int = 0  # 정답을 하나도 못 찾은 질문 수

    def add(self, recall: float, rr: float) -> None:
        self.recalls.append(recall)
        self.rrs.append(rr)
        self.zero_hit += recall == 0

    @property
    def n(self) -> int:
        return len(self.recalls)

    @property
    def recall(self) -> float:
        return mean(self.recalls) if self.recalls else float("nan")

    @property
    def mrr(self) -> float:
        return mean(self.rrs) if self.rrs else float("nan")


def _first(record: dict, candidates: Sequence[str]):
    return next((record[c] for c in candidates if c in record), None)


def _norm_lang(value) -> str:
    v = str(value or "").lower()
    return "ko" if v.startswith(("ko", "kr", "한")) else "en" if v.startswith(("en", "영")) else v


def _doc_ids(value) -> list[str]:
    """["A", "B"] / "A" / [{"doc_id": "A", "page": 3}] 등 다양한 표기를 문헌 ID 목록으로"""
    if value is None:
        return []
    if isinstance(value, (str, int)):
        return [str(value)]
    out = []
    for v in value:
        out.append(str(v.get("doc_id") or v.get("doc") or v.get("id")) if isinstance(v, dict) else str(v))
    return out


def _pages(value) -> list[tuple[str, int]]:
    out = []
    for v in value or []:
        if isinstance(v, dict) and v.get("page") is not None:
            doc = v.get("doc_id") or v.get("doc") or v.get("id")
            pages = v["page"] if isinstance(v["page"], list) else [v["page"]]
            out.extend((str(doc), int(p)) for p in pages)
    return out


def load_questions(path: str | Path) -> list[EvalQuestion]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else next(raw[k] for k in ("questions", "items", "data") if k in raw)
    out = []
    for i, r in enumerate(records):
        question = _first(r, QUESTION_FIELDS)
        if question is None:
            raise KeyError(f"질문 필드를 찾지 못했습니다. 사용 가능한 키: {sorted(r)}")
        gold = _first(r, GOLD_DOC_FIELDS)
        evidence = _first(r, EVIDENCE_FIELDS)
        answerable = r.get("answerable", True)
        docs = _doc_ids(gold) or _doc_ids(evidence)
        out.append(
            EvalQuestion(
                qid=str(_first(r, ID_FIELDS) or i),
                question=str(question),
                lang=_norm_lang(_first(r, LANG_FIELDS)),
                split=str(_first(r, SPLIT_FIELDS) or "dev").lower(),
                gold_docs=frozenset(docs) if answerable else frozenset(),
                gold_pages=frozenset(_pages(evidence) or _pages(gold)) if answerable else frozenset(),
            )
        )
    return out


def score_one(
    retrieved: Sequence[tuple[str, int | None]], q: EvalQuestion, unit: Unit = "doc", k: int = 8
) -> tuple[float, float]:
    """retrieved: 순위대로 (문헌 ID, 페이지). 반환: (recall@k, 역순위)"""
    top = list(retrieved)[:k]
    if unit == "page":
        gold = q.gold_pages
        keys = [(d, p) for d, p in top]
    else:
        gold = q.gold_docs
        keys = [d for d, _ in top]
    if not gold:
        raise ValueError(f"{q.qid}: 정답이 없는 질문은 채점하지 않습니다(unit={unit}).")
    found = {key for key in keys if key in gold}
    rr = next((1.0 / r for r, key in enumerate(keys, start=1) if key in gold), 0.0)
    return len(found) / len(gold), rr


def aggregate(rows: Iterable[tuple[EvalQuestion, float, float]]) -> dict[tuple[str, str], Scores]:
    """(split, lang) 조합별 + 'all' 합계. split ∈ {dev, test, all}, lang ∈ {ko, en, all}"""
    table: dict[tuple[str, str], Scores] = {}
    for q, recall, rr in rows:
        for split in (q.split, "all"):
            for lang in (q.lang, "all"):
                table.setdefault((split, lang), Scores()).add(recall, rr)
    return table
