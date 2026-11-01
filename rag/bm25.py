"""BM25 키워드 검색.

⚠ 임시 구현: 기존 앱의 BM25(토크나이저·한국어 용어 목록 확장 포함) 코드를 받으면 그것으로 바꿔야 한다.
기준선(전체 recall@8 0.73 / MRR 0.63)과 공정하게 비교하려면 하이브리드의 BM25 쪽도 기존 구현과 같아야 하기 때문이다.
교체 방법: KeywordSearcher 프로토콜(search(query, top_n) -> list[KeywordHit])만 맞춘 어댑터를 만들어 넘기면 된다.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Protocol

from rag.corpus import Paragraph

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*|[가-힣]+")


@dataclass(frozen=True)
class KeywordHit:
    paragraph_id: str
    doc_id: str
    score: float


class KeywordSearcher(Protocol):
    def search(self, query: str, top_n: int) -> list[KeywordHit]: ...


def tokenize(text: str) -> list[str]:
    """규격명은 전체("iapws-if97")와 부분("iapws", "if97")을 모두 토큰으로 — "IF97"만 검색해도 찾게"""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if "-" in tok or "." in tok:
            out.extend(part for part in re.split(r"[-.]", tok) if part)
    return out


class BM25Searcher:
    def __init__(
        self,
        paragraphs: list[Paragraph],
        k1: float = 1.5,
        b: float = 0.75,
        query_expander: Callable[[str], str] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
    ):
        """tokenizer: 기본은 영숫자 토크나이저. RAG API는 rag.tokenizer.tokenize(한국어 형태소 + 보호 토큰)를 넘긴다."""
        self._paragraphs = paragraphs
        self._tokenize = tokenizer or tokenize
        self._k1, self._b = k1, b
        self._expand = query_expander or (lambda q: q)
        self._tf = [Counter(self._tokenize(p.text)) for p in paragraphs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(len(self._len), 1)
        df = Counter(t for tf in self._tf for t in tf)
        n = len(paragraphs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, top_n: int) -> list[KeywordHit]:
        terms = [t for t in self._tokenize(self._expand(query)) if t in self._idf]
        if not terms:
            return []
        scores: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            norm = self._k1 * (1 - self._b + self._b * self._len[i] / self._avg)
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self._idf[t] * f * (self._k1 + 1) / (f + norm)
            if s > 0:
                scores.append((s, i))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [
            KeywordHit(self._paragraphs[i].paragraph_id, self._paragraphs[i].doc_id, s) for s, i in scores[:top_n]
        ]


def glossary_expander(glossary: dict[str, list[str] | str]) -> Callable[[str], str]:
    """기존 앱 방식: 한국어 질문에 용어 목록의 영어 단어를 덧붙인다 (예: 과열기 → superheater)."""

    def expand(query: str) -> str:
        extra = []
        for ko, en in glossary.items():
            if ko in query:
                extra.extend([en] if isinstance(en, str) else en)
        return f"{query} {' '.join(extra)}" if extra else query

    return expand
