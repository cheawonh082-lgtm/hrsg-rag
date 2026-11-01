"""임베딩 모델. 기본은 로컬 BAAI/bge-m3 (밀집 벡터, 1,024차원, L2 정규화).

- BGEM3Embedder: FlagEmbedding이 있으면 그것을, 없으면 sentence-transformers로 같은 모델을 불러온다.
  무거운 라이브러리(torch)는 실제로 임베딩할 때만 import 한다.
- Qwen3Embedder: Qwen/Qwen3-Embedding (크기는 설정값). 질의에만 instruction 프롬프트를 붙이고 문서에는 붙이지 않는다.
- HashingEmbedder: 테스트 전용. 모델 없이 결정적으로 동작하며 품질 평가에는 쓰지 않는다.

embed()는 문서용, embed_queries()는 질의용이다. BGE-M3는 둘이 같다(지시문 없음).
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

MODEL_ID = "BAAI/bge-m3"
DIM = 1024


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """문서 임베딩. (len(texts), dim) float32, 행마다 L2 정규화 — 내적 = 코사인 유사도"""
        ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """질의 임베딩 (모델에 따라 instruction을 붙임)"""
        ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class BGEM3Embedder:
    name = MODEL_ID
    dim = DIM

    def __init__(self, model_id: str = MODEL_ID, batch_size: int = 16, max_length: int = 8192, device: str | None = None):
        self.name = model_id
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = device
        self._model = None
        self._backend = ""

    def _load(self):
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(self.name, use_fp16=False, devices=self._device)
            self._backend = "FlagEmbedding"
        except ImportError:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "BGE-M3를 실행하려면 `pip install -r requirements-embed.txt`로 FlagEmbedding 또는 sentence-transformers를 설치하세요."
                ) from e
            self._model = SentenceTransformer(self.name, device=self._device)
            self._model.max_seq_length = self._max_length
            self._backend = "sentence-transformers"

    @property
    def backend(self) -> str:
        self._load()
        return self._backend

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        if self._backend == "FlagEmbedding":
            out = self._model.encode(list(texts), batch_size=self._batch_size, max_length=self._max_length)["dense_vecs"]
        else:
            out = self._model.encode(list(texts), batch_size=self._batch_size, normalize_embeddings=True)
        return normalize(out)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed(texts)


QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN_INSTRUCTION = (
    "Given a question about heat recovery steam generators and combined cycle power plants, "
    "retrieve passages from technical papers and reports that answer the question"
)


class Qwen3Embedder:
    """Qwen3-Embedding (0.6B / 4B / 8B). sentence-transformers 설정(마지막 토큰 풀링, 왼쪽 패딩)을 그대로 쓴다.
    질의: "Instruct: {instruction}\\nQuery:{질의}" — 모델 카드 권장대로 instruction은 영어로 둔다(질의 언어와 무관).
    문서: 원문 그대로."""

    def __init__(self, model_id: str = QWEN_MODEL_ID, instruction: str = QWEN_INSTRUCTION, batch_size: int = 8,
                 max_length: int = 1024, device: str | None = None):
        self.name = model_id
        self.instruction = instruction
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = device
        self._model = None
        self.backend = "sentence-transformers"

    @property
    def dim(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()

    @property
    def query_prompt(self) -> str:
        return f"Instruct: {self.instruction}\nQuery:"

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            # 가중치는 bfloat16으로 배포되지만 CPU에서는 float32가 훨씬 빠르다
            on_gpu = (self._device or "").startswith("cuda") or (self._device is None and torch.cuda.is_available())
            kwargs = {} if on_gpu else {"model_kwargs": {"dtype": torch.float32}}
            self._model = SentenceTransformer(self.name, device=self._device, **kwargs)
            self._model.max_seq_length = self._max_length

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        return normalize(self._model.encode(list(texts), batch_size=self._batch_size, normalize_embeddings=True))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        return normalize(self._model.encode(list(texts), prompt=self.query_prompt, batch_size=self._batch_size,
                                            normalize_embeddings=True))


class HashingEmbedder:
    """테스트용: 토큰 해시를 차원에 누적한 bag-of-words 벡터"""

    name = "hashing-test"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return normalize(out)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed(texts)
