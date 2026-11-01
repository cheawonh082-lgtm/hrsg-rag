"""하이브리드 검색: 벡터(BGE-M3) 상위 N + BM25 상위 N → RRF 결합 → 문헌당 최대 2문단 → 최종 8개.

RRF(d) = Σ 1 / (k + rank(d))   (rank는 1부터, 각 검색 방식에 대해 합산, 기본 k=60)
예: 벡터 1위 + BM25 3위 → 1/61 + 1/63 ≈ 0.0323, 벡터 2위만 → 1/62 ≈ 0.0161
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from rag.bm25 import KeywordSearcher
from rag.embedder import Embedder
from rag.vector_index import LocalVectorIndex

DEFAULT_TOP_N = 40
DEFAULT_RRF_K = 60
DEFAULT_PER_DOC = 2
DEFAULT_FINAL_K = 8


@dataclass(frozen=True)
class SearchHit:
    paragraph_id: str
    doc_id: str
    score: float  # RRF 점수 (단일 방식 검색이면 그 방식의 점수)
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None  # 코사인 유사도 — '자료 없음' 판단 참고용


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = DEFAULT_RRF_K) -> list[tuple[str, float]]:
    """여러 순위 목록(문단 ID, 1위부터)을 RRF로 합친다. 동점이면 먼저 나온 목록의 순서를 따른다."""
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, tuple[int, int]] = {}
    for li, ranking in enumerate(rankings):
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] += 1.0 / (k + rank)
            first_seen.setdefault(pid, (rank, li))
    return sorted(scores.items(), key=lambda x: (-x[1], first_seen[x[0]]))


def diversify(
    hits: Iterable[SearchHit], per_doc: int = DEFAULT_PER_DOC, final_k: int = DEFAULT_FINAL_K,
    doc_group: dict[str, str] | None = None,
) -> list[SearchHit]:
    """점수 순서를 유지하면서 문헌당 per_doc개까지만 담아 final_k개를 고른다.
    doc_group: 문헌 ID → 묶음 ID (예: EN/KO 병렬 문서의 pair_id). 주면 묶음 단위로 per_doc개까지만 담는다."""
    taken: dict[str, int] = defaultdict(int)
    out: list[SearchHit] = []
    for h in hits:
        key = doc_group.get(h.doc_id, h.doc_id) if doc_group else h.doc_id
        if taken[key] >= per_doc:
            continue
        taken[key] += 1
        out.append(h)
        if len(out) == final_k:
            break
    return out


class HybridSearcher:
    def __init__(
        self,
        index: LocalVectorIndex,
        embedder: Embedder,
        bm25: KeywordSearcher,
        top_n: int = DEFAULT_TOP_N,
        rrf_k: int = DEFAULT_RRF_K,
        per_doc: int = DEFAULT_PER_DOC,
        final_k: int = DEFAULT_FINAL_K,
        lang_groups: dict[str, list[str]] | None = None,
        doc_group: dict[str, str] | None = None,
    ):
        """lang_groups: {"ko": [문단 ID…], "en": [...]} — 주면 벡터 검색을 언어별로 따로 해 RRF에 각각 넣는다(언어 균형).
        BGE-M3는 질문과 같은 언어의 문단을 더 높게 매기는 경향이 있어, 한국어 문헌이 섞이면 한국어 질문에서
        영어 정답 문헌이 밀려난다. 언어 균형은 재현율을 높이지만 MRR은 낮아질 수 있다(기본 꺼짐).
        doc_group: {문헌 ID: 묶음 ID} — 주면 문헌당 제한을 묶음(pair_id) 단위로 적용한다(EN/KO 쌍둥이 중복 제거)."""
        self.index, self.embedder, self.bm25 = index, embedder, bm25
        self.top_n, self.rrf_k, self.per_doc, self.final_k = top_n, rrf_k, per_doc, final_k
        self.lang_groups = lang_groups
        self.doc_group = doc_group
        self._doc_of = dict(zip(index.paragraph_ids, index.doc_ids))
        self._masks = None
        if lang_groups:
            pos = {pid: i for i, pid in enumerate(index.paragraph_ids)}
            self._masks = []
            for ids in lang_groups.values():
                mask = np.zeros(len(pos), dtype=bool)
                mask[[pos[i] for i in ids if i in pos]] = True
                self._masks.append(mask)

    def with_params(self, **params) -> "HybridSearcher":
        cfg = dict(top_n=self.top_n, rrf_k=self.rrf_k, per_doc=self.per_doc, final_k=self.final_k, lang_groups=self.lang_groups, doc_group=self.doc_group) | params
        return HybridSearcher(self.index, self.embedder, self.bm25, **cfg)

    def _vector_rankings(self, query_vector) -> list[list]:
        if self._masks is None:
            return [self.index.search(query_vector, self.top_n)]
        return [self.index.search(query_vector, self.top_n, mask=m) for m in self._masks]

    def search(self, query: str, query_vector=None) -> list[SearchHit]:
        """query_vector를 넘기면 임베딩을 다시 하지 않는다(평가 시 질문 벡터 캐시용)."""
        if query_vector is None:
            query_vector = self.embedder.embed([query])[0]
        vec_lists = self._vector_rankings(query_vector)
        kw_hits = self.bm25.search(query, self.top_n) if self.bm25 is not None else []

        vec_rank: dict[str, int] = {}
        vec_score: dict[str, float] = {}
        for hits in vec_lists:
            for r, h in enumerate(hits, start=1):
                vec_rank[h.paragraph_id] = min(r, vec_rank.get(h.paragraph_id, r))
                vec_score[h.paragraph_id] = h.score
        kw_rank = {h.paragraph_id: r for r, h in enumerate(kw_hits, start=1)}
        rankings = [[h.paragraph_id for h in hits] for hits in vec_lists]
        if kw_hits:
            rankings.append([h.paragraph_id for h in kw_hits])
        fused = rrf_fuse(rankings, self.rrf_k)

        hits = (
            SearchHit(pid, self._doc_of[pid], score, vec_rank.get(pid), kw_rank.get(pid), vec_score.get(pid))
            for pid, score in fused
        )
        return diversify(hits, self.per_doc, self.final_k, self.doc_group)

    def search_vector_only(self, query: str, query_vector=None) -> list[SearchHit]:
        if query_vector is None:
            query_vector = self.embedder.embed([query])[0]
        lists = self._vector_rankings(query_vector)
        if len(lists) == 1:
            hits = (SearchHit(h.paragraph_id, h.doc_id, h.score, r, None, h.score) for r, h in enumerate(lists[0], start=1))
            return diversify(hits, self.per_doc, self.final_k, self.doc_group)
        score = {h.paragraph_id: h.score for hits in lists for h in hits}
        fused = rrf_fuse([[h.paragraph_id for h in hits] for hits in lists], self.rrf_k)
        return diversify((SearchHit(pid, self._doc_of[pid], s, None, None, score[pid]) for pid, s in fused), self.per_doc, self.final_k, self.doc_group)


def bm25_only(bm25: KeywordSearcher, query: str, final_k: int = DEFAULT_FINAL_K, per_doc: int | None = None, top_n: int = 50) -> list[SearchHit]:
    """기준선: BM25 상위 결과. per_doc=None이면 다양성 조건 없이 상위 final_k개 (기존 앱 방식에 맞춰 선택)."""
    hits = [SearchHit(h.paragraph_id, h.doc_id, h.score, None, r) for r, h in enumerate(bm25.search(query, top_n), start=1)]
    return diversify(hits, per_doc, final_k) if per_doc else hits[:final_k]
