"""임베딩 캐시 — (모델, 최대 길이, 텍스트) 해시 → 벡터.

코퍼스 구성(mixed/ko/en_strict)끼리 같은 청크가 많고, 논문이 추가돼도 기존 청크는 그대로라
이미 임베딩한 텍스트는 다시 계산하지 않는다. 저장: <dir>/vectors.npy + keys.json (같은 순서).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def text_key(model: str, max_length: int, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{max_length}\x00{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self._keys: list[str] = []
        self._vecs: np.ndarray | None = None
        if (self.dir / "keys.json").exists():
            self._keys = json.loads((self.dir / "keys.json").read_text(encoding="utf-8"))
            self._vecs = np.load(self.dir / "vectors.npy")
        self._pos = {k: i for i, k in enumerate(self._keys)}

    def __len__(self) -> int:
        return len(self._keys)

    def get(self, key: str) -> np.ndarray | None:
        i = self._pos.get(key)
        return None if i is None else self._vecs[i]

    def add(self, keys: list[str], vectors: np.ndarray) -> None:
        new = [(k, v) for k, v in zip(keys, vectors) if k not in self._pos]
        if not new:
            return
        stacked = np.asarray([v for _, v in new], dtype=np.float32)
        self._vecs = stacked if self._vecs is None else np.vstack([self._vecs, stacked])
        for k, _ in new:
            self._pos[k] = len(self._keys)
            self._keys.append(k)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if self._vecs is not None:
            np.save(self.dir / "vectors.npy", self._vecs)
        (self.dir / "keys.json").write_text(json.dumps(self._keys), encoding="utf-8")
