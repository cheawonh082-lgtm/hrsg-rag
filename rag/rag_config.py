"""RAG API 설정 — 환경변수(.env 포함)로 바꾼다. 값이 없으면 아래 기본값."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from rag.embedder import MODEL_ID as BGE_MODEL_ID
from rag.embedder import QWEN_INSTRUCTION, QWEN_MODEL_ID
from rag.env import load_dotenv
from rag.lang import KO_RATIO

ROOT = Path(__file__).resolve().parents[1]


def parse_weights(text: str) -> dict[str, float]:
    """"bge=1,qwen=1,bm25=0.5" → {"bge": 1.0, ...}"""
    out = {}
    for part in text.split(","):
        if part.strip():
            key, value = part.split("=")
            out[key.strip()] = float(value)
    return out


def model_slug(model_id: str) -> str:
    """인덱스 폴더 이름 — 모델(크기)이 바뀌면 다른 폴더를 쓰게 한다. Qwen/Qwen3-Embedding-0.6B → qwen3-embedding-0.6b"""
    return model_id.rstrip("/").split("/")[-1].lower()


@dataclass
class RagSettings:
    corpus: Path = ROOT / "data/rag/corpus_all.json"
    index_dir: Path = ROOT / "index/rag"
    cache_dir: Path = ROOT / "index/cache"
    bge_model: str = BGE_MODEL_ID
    qwen_model: str = QWEN_MODEL_ID
    qwen_instruction: str = QWEN_INSTRUCTION
    max_length: int = 1024          # 문단 최대 토큰 (가장 긴 문단 약 360단어 — 잘리지 않음)
    top_k: int = 50                 # 검색기마다 가져오는 후보 수
    top_n: int = 5                  # 최종 결과 수
    rrf_k: int = 60
    ko_ratio: float = KO_RATIO      # 질문 언어 판별: 한글 비율이 이 값 이상이면 ko
    # 4안: 질문 언어별 검색기 가중치. 한국어 질문은 정답 대부분이 영어 문단이라 BM25가 엉뚱한 한국어 문단을
    # 올리기만 해서 끈다 (용어 확장 후에도 0이 가장 좋았다 — eval/RESULTS.md).
    weights_ko: dict[str, float] = field(default_factory=lambda: {"bge": 1.0, "qwen": 1.0, "bm25": 0.0})
    weights_en: dict[str, float] = field(default_factory=lambda: {"bge": 1.0, "qwen": 1.0, "bm25": 1.0})
    expand_ko: bool = True          # 한국어 질문에 한↔영 용어 사전(rag.glossary)의 영어 용어를 덧붙여 검색
    device: str | None = None
    warmup: bool = False            # 서버 시작 때 모델을 미리 불러올지

    @property
    def bge_index(self) -> Path:
        return self.index_dir / model_slug(self.bge_model)

    @property
    def qwen_index(self) -> Path:
        return self.index_dir / model_slug(self.qwen_model)

    @classmethod
    def from_env(cls) -> "RagSettings":
        load_dotenv()
        s, env = cls(), os.environ
        path = lambda v: Path(v) if Path(v).is_absolute() else ROOT / v  # noqa: E731
        if v := env.get("RAG_CORPUS"):
            s.corpus = path(v)
        if v := env.get("RAG_INDEX_DIR"):
            s.index_dir = path(v)
        if v := env.get("RAG_CACHE_DIR"):
            s.cache_dir = path(v)
        s.bge_model = env.get("RAG_BGE_MODEL", s.bge_model)
        s.qwen_model = env.get("RAG_QWEN_MODEL", s.qwen_model)
        s.qwen_instruction = env.get("RAG_QWEN_INSTRUCTION", s.qwen_instruction)
        s.max_length = int(env.get("RAG_MAX_LENGTH", s.max_length))
        s.top_k = int(env.get("RAG_TOP_K", s.top_k))
        s.top_n = int(env.get("RAG_TOP_N", s.top_n))
        s.rrf_k = int(env.get("RAG_RRF_K", s.rrf_k))
        s.ko_ratio = float(env.get("RAG_KO_RATIO", s.ko_ratio))
        if v := env.get("RAG_WEIGHTS_KO"):
            s.weights_ko = parse_weights(v)
        if v := env.get("RAG_WEIGHTS_EN"):
            s.weights_en = parse_weights(v)
        s.expand_ko = env.get("RAG_EXPAND_KO", "1").lower() not in ("0", "false", "no")
        s.device = env.get("RAG_DEVICE") or None
        s.warmup = env.get("RAG_WARMUP", "").lower() in ("1", "true", "yes")
        return s
