"""로컬 벡터 인덱스 — 정규화된 벡터의 내적(= 코사인 유사도)으로 전수 검색.

1,800개 문단 × 1,024차원은 numpy 행렬 하나로 충분히 빠르다(별도 벡터 DB 불필요).
저장 형식: <dir>/embeddings.npy (float32) + <dir>/meta.json (문단·문헌 ID, 모델, 코퍼스 버전)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.corpus import Paragraph


@dataclass(frozen=True)
class VectorHit:
    paragraph_id: str
    doc_id: str
    score: float  # 코사인 유사도


class LocalVectorIndex:
    def __init__(self, vectors: np.ndarray, paragraph_ids: list[str], doc_ids: list[str], meta: dict | None = None):
        if len(vectors) != len(paragraph_ids) or len(paragraph_ids) != len(doc_ids):
            raise ValueError("벡터 수와 ID 수가 다릅니다.")
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.paragraph_ids = paragraph_ids
        self.doc_ids = doc_ids
        self.meta = meta or {}

    @classmethod
    def build(cls, paragraphs: list[Paragraph], vectors: np.ndarray, meta: dict | None = None) -> "LocalVectorIndex":
        return cls(vectors, [p.paragraph_id for p in paragraphs], [p.doc_id for p in paragraphs], meta)

    def search(self, query_vector: np.ndarray, top_n: int, mask: np.ndarray | None = None) -> list[VectorHit]:
        """mask: 검색 대상 문단만 True (예: 언어별 검색)"""
        scores = self.vectors @ np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
            top_n = min(top_n, int(mask.sum()))
        top_n = min(top_n, len(scores))
        if top_n <= 0:
            return []
        idx = np.argpartition(-scores, top_n - 1)[:top_n]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [VectorHit(self.paragraph_ids[i], self.doc_ids[i], float(scores[i])) for i in idx]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.vectors)
        meta = {**self.meta, "count": len(self.paragraph_ids), "dim": int(self.vectors.shape[1])}
        (d / "meta.json").write_text(
            json.dumps({**meta, "paragraph_ids": self.paragraph_ids, "doc_ids": self.doc_ids}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "LocalVectorIndex":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        ids, docs = meta.pop("paragraph_ids"), meta.pop("doc_ids")
        return cls(np.load(d / "embeddings.npy"), ids, docs, meta)
