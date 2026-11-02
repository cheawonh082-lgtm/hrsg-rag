"""RAG 검색 엔진 — 검색기 3개(bge, qwen, bm25)와 검색 방식 4가지.

검색기 (각 top_k개, 기본 50)
- bge  : BAAI/bge-m3 dense, 정규화 벡터의 코사인 유사도
- qwen : Qwen3-Embedding dense (질의에만 instruction)
- bm25 : section + text + terms, 한국어 형태소 + 보호 토큰 (rag.tokenizer)

방식 (최종 top_n개, 기본 5)
- 1안: bge + bm25 → RRF
- 2안: qwen + bm25 → RRF
- 3안: bge + qwen + bm25 → RRF
- 4안: 3안 검색기 → 질문 언어(한글 비율)별 가중 RRF   score(d) = Σ w_r / (k + rank_r(d))

한국어 질문은 모든 방식에서 한↔영 용어 사전으로 영어 용어를 덧붙인 질의로 검색한다 (RAG_EXPAND_KO, rag.glossary).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from rag.bm25 import BM25Searcher
from rag.corpus import Paragraph
from rag.corpus_all import Doc, Para, load_corpus_all
from rag.embedder import BGEM3Embedder, Qwen3Embedder
from rag.glossary import expand_query
from rag.lang import detect_lang
from rag.rag_config import RagSettings
from rag.tokenizer import tokenize
from rag.vector_index import LocalVectorIndex

RETRIEVERS = ("bge", "qwen", "bm25")
METHODS: dict[str, dict] = {
    "1": {"retrievers": ("bge", "bm25"), "weighted": False, "label": "bge + BM25 → RRF"},
    "2": {"retrievers": ("qwen", "bm25"), "weighted": False, "label": "qwen + BM25 → RRF"},
    "3": {"retrievers": ("bge", "qwen", "bm25"), "weighted": False, "label": "bge + qwen + BM25 → RRF"},
    "4": {"retrievers": ("bge", "qwen", "bm25"), "weighted": True, "label": "bge + qwen + BM25 → 질문 언어별 가중 RRF"},
}


class IndexUnavailable(RuntimeError):
    """요청한 방식에 필요한 인덱스가 아직 없다 (scripts/build_rag_index.py로 만든다)."""


def weighted_rrf(rankings: dict[str, list[str]], k: int, weights: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """rankings: {검색기: [문단 ID, 1위부터]}. 가중치가 없으면 모두 1 (= 일반 RRF).
    동점이면 어느 검색기에서든 더 높은 순위였던 문단을 앞에 둔다."""
    score: dict[str, float] = defaultdict(float)
    best: dict[str, int] = {}
    for name, ranking in rankings.items():
        w = 1.0 if weights is None else weights.get(name, 0.0)
        for rank, pid in enumerate(ranking, start=1):
            score[pid] += w / (k + rank)
            best[pid] = min(rank, best.get(pid, rank))
    return sorted(score.items(), key=lambda x: (-x[1], best[x[0]], x[0]))


@dataclass
class Hit:
    id: str
    score: float


class DenseRetriever:
    def __init__(self, name: str, index: LocalVectorIndex, embedder):
        self.name, self.index, self.embedder = name, index, embedder
        self._lock = threading.Lock()  # 모델 encode는 한 번에 하나씩

    def query_vector(self, query: str) -> np.ndarray:
        with self._lock:
            return self.embedder.embed_queries([query])[0]

    def search(self, query: str, top_k: int) -> list[Hit]:
        return [Hit(h.paragraph_id, h.score) for h in self.index.search(self.query_vector(query), top_k)]


class BM25Retriever:
    name = "bm25"

    def __init__(self, paras: list[Para]):
        self._bm25 = BM25Searcher([Paragraph(p.id, p.doc_id, p.bm25_input) for p in paras], tokenizer=tokenize)

    def search(self, query: str, top_k: int) -> list[Hit]:
        return [Hit(h.paragraph_id, h.score) for h in self._bm25.search(query, top_k)]


class RagEngine:
    def __init__(self, settings: RagSettings, embedders: dict | None = None):
        """embedders: {"bge": ..., "qwen": ...} — 테스트에서 가짜 임베더를 넣을 때만 쓴다."""
        self.settings = settings
        self.docs: dict[str, Doc]
        self.docs, self.paras = load_corpus_all(settings.corpus)
        self.para_by_id = {p.id: p for p in self.paras}
        self.retrievers: dict[str, object] = {"bm25": BM25Retriever(self.paras)}
        self.missing: dict[str, str] = {}
        embedders = embedders or {}
        for name, path, make in (
            ("bge", settings.bge_index, lambda: BGEM3Embedder(settings.bge_model, max_length=settings.max_length, device=settings.device)),
            ("qwen", settings.qwen_index, lambda: Qwen3Embedder(settings.qwen_model, settings.qwen_instruction,
                                                                max_length=settings.max_length, device=settings.device)),
        ):
            if not (path / "meta.json").exists():
                self.missing[name] = f"{path} 없음 — scripts/build_rag_index.py로 만드세요"
                continue
            index = LocalVectorIndex.load(path)
            if index.paragraph_ids != [p.id for p in self.paras]:
                self.missing[name] = f"{path}가 현재 코퍼스와 맞지 않습니다 — 인덱스를 다시 만드세요"
                continue
            self.retrievers[name] = DenseRetriever(name, index, embedders.get(name) or make())

    # ---- 정보 ----
    def available_methods(self) -> dict[str, bool]:
        return {m: all(r in self.retrievers for r in spec["retrievers"]) for m, spec in METHODS.items()}

    def weights_for(self, lang: str) -> dict[str, float]:
        return self.settings.weights_ko if lang == "ko" else self.settings.weights_en

    def warmup(self) -> None:
        for r in self.retrievers.values():
            if isinstance(r, DenseRetriever):
                r.query_vector("warmup")

    # ---- 검색 ----
    def search_query(self, query: str) -> str:
        """검색기에 실제로 넣는 질의 — 한국어 질문이면 영어 용어를 덧붙인다"""
        if self.settings.expand_ko and detect_lang(query, self.settings.ko_ratio) == "ko":
            return expand_query(query)
        return query

    def retrieve(self, query: str, names: tuple[str, ...], top_k: int) -> tuple[dict[str, list[Hit]], dict[str, float]]:
        """검색기별 결과와 걸린 시간(ms)"""
        query = self.search_query(query)
        hits, timings = {}, {}
        for name in names:
            if name not in self.retrievers:
                raise IndexUnavailable(f"'{name}' 검색기를 쓸 수 없습니다: {self.missing.get(name, '알 수 없는 검색기')}")
            started = time.perf_counter()
            hits[name] = self.retrievers[name].search(query, top_k)
            timings[name] = round((time.perf_counter() - started) * 1000, 1)
        return hits, timings

    def search(self, query: str, method: str = "4", top_n: int | None = None, top_k: int | None = None,
               _hits: dict[str, list[Hit]] | None = None) -> dict:
        if method not in METHODS:
            raise ValueError(f"method는 {list(METHODS)} 중 하나여야 합니다")
        spec = METHODS[method]
        top_n = top_n or self.settings.top_n
        top_k = top_k or self.settings.top_k
        lang = detect_lang(query, self.settings.ko_ratio)
        if _hits is None:
            hits, timings = self.retrieve(query, spec["retrievers"], top_k)
        else:  # compare: 검색기 결과를 방식끼리 공유
            missing = [r for r in spec["retrievers"] if r not in _hits]
            if missing:
                raise IndexUnavailable(f"{missing} 검색기를 쓸 수 없습니다: " + "; ".join(self.missing.get(m, "") for m in missing))
            hits, timings = {r: _hits[r] for r in spec["retrievers"]}, {}
        weights = self.weights_for(lang) if spec["weighted"] else None
        fused = weighted_rrf({r: [h.id for h in hs] for r, hs in hits.items()}, self.settings.rrf_k, weights)
        ranks = {r: {h.id: (i, h.score) for i, h in enumerate(hs, start=1)} for r, hs in hits.items()}
        results = []
        for rank, (pid, score) in enumerate(fused[:top_n], start=1):
            p, d = self.para_by_id[pid], self.docs[self.para_by_id[pid].doc_id]
            results.append({
                "rank": rank, "id": p.id, "doc_id": p.doc_id, "title": d.title, "section": p.section,
                "page_start": p.page_start, "page_end": p.page_end, "content_type": p.content_type,
                "source_type": p.source_type, "lang": p.lang, "doc_lang": d.lang, "text": p.text, "terms": list(p.terms),
                "score": round(score, 6),
                "retrievers": {r: {"rank": ranks[r][pid][0], "score": round(float(ranks[r][pid][1]), 4)}
                               for r in hits if pid in ranks[r]},
            })
        return {"query": query, "query_lang": lang, "search_query": self.search_query(query), "method": method, "method_label": spec["label"],
                "weights": weights or {r: 1.0 for r in spec["retrievers"]}, "top_k": top_k, "top_n": top_n,
                "timings_ms": timings, "results": results}

    def compare(self, query: str, top_n: int | None = None, top_k: int | None = None) -> dict:
        """1~4안을 한 번에 — 검색기는 한 번씩만 돌린다. 쓸 수 없는 방식은 error로 표시."""
        top_k = top_k or self.settings.top_k
        names = tuple(r for r in RETRIEVERS if r in self.retrievers)
        hits, timings = self.retrieve(query, names, top_k)
        out = {}
        for m in METHODS:
            try:
                out[m] = self.search(query, m, top_n, top_k, _hits=hits)
            except IndexUnavailable as e:
                out[m] = {"method": m, "error": str(e)}
        return {"query": query, "query_lang": detect_lang(query, self.settings.ko_ratio), "timings_ms": timings, "methods": out}
