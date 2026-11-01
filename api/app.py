"""HRSG RAG 검색 API (FastAPI).

    .venv/Scripts/python -m uvicorn api.app:app --port 8100

엔드포인트
- GET  /health                 코퍼스·인덱스 상태, 쓸 수 있는 방식
- GET  /methods                1~4안 설명과 4안 언어별 가중치
- POST /search                 {"query", "method": "1"~"4", "top_n", "top_k"} → 상위 문단
- POST /search/compare         {"query", "top_n", "top_k"} → 1~4안 결과를 한 번에 (검색기는 한 번씩만 실행)
- GET  /paragraphs/{id}        문단 하나
- GET  /documents/{doc_id}     문헌 정보 (출처 표시용)

설정은 환경변수(RAG_*)로 바꾼다 — rag/rag_config.py 참고. 생성(LLM 답변)은 이 API 범위 밖이다.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.rag_config import RagSettings  # noqa: E402
from rag.rag_engine import METHODS, IndexUnavailable, RagEngine  # noqa: E402

Method = Literal["1", "2", "3", "4"]
_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    if _engine is None:
        raise HTTPException(503, "검색 엔진이 아직 준비되지 않았습니다")
    return _engine


def set_engine(engine: RagEngine | None) -> None:
    """테스트용: 미리 만든 엔진을 넣는다."""
    global _engine
    _engine = engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _engine is None:
        settings = RagSettings.from_env()
        set_engine(RagEngine(settings))
        if settings.warmup:
            _engine.warmup()
    yield


app = FastAPI(title="HRSG RAG API", version="1.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    method: Method = "4"
    top_n: int | None = Field(default=None, ge=1, le=50, description="최종 결과 수 (기본 RAG_TOP_N=5)")
    top_k: int | None = Field(default=None, ge=1, le=500, description="검색기별 후보 수 (기본 RAG_TOP_K=50)")


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_n: int | None = Field(default=None, ge=1, le=50)
    top_k: int | None = Field(default=None, ge=1, le=500)


@app.get("/health")
def health() -> dict:
    e = get_engine()
    s = e.settings
    return {
        "status": "ok",
        "corpus": {"path": str(s.corpus), "documents": len(e.docs), "paragraphs": len(e.paras)},
        "retrievers": {name: name in e.retrievers for name in ("bge", "qwen", "bm25")},
        "missing": e.missing,
        "methods": e.available_methods(),
        "models": {"bge": s.bge_model, "qwen": s.qwen_model},
    }


@app.get("/methods")
def methods() -> dict:
    e = get_engine()
    s = e.settings
    return {
        "methods": {m: {"label": spec["label"], "retrievers": list(spec["retrievers"]), "weighted": spec["weighted"],
                        "available": e.available_methods()[m]} for m, spec in METHODS.items()},
        "weights": {"ko": s.weights_ko, "en": s.weights_en},
        "lang_detection": {"rule": "한글 / (한글 + 라틴 문자) ≥ ko_ratio → ko", "ko_ratio": s.ko_ratio},
        "defaults": {"top_k": s.top_k, "top_n": s.top_n, "rrf_k": s.rrf_k},
        "qwen_instruction": s.qwen_instruction,
    }


@app.post("/search")
def search(req: SearchRequest) -> dict:
    try:
        return get_engine().search(req.query, req.method, req.top_n, req.top_k)
    except IndexUnavailable as e:
        raise HTTPException(503, str(e)) from e


@app.post("/search/compare")
def compare(req: CompareRequest) -> dict:
    return get_engine().compare(req.query, req.top_n, req.top_k)


@app.get("/paragraphs/{pid}")
def paragraph(pid: str) -> dict:
    e = get_engine()
    p = e.para_by_id.get(pid)
    if p is None:
        raise HTTPException(404, f"문단 {pid}가 없습니다")
    return {**asdict(p), "title": e.docs[p.doc_id].title}


@app.get("/documents/{doc_id}")
def document(doc_id: str) -> dict:
    e = get_engine()
    d = e.docs.get(doc_id)
    if d is None:
        raise HTTPException(404, f"문헌 {doc_id}가 없습니다")
    return {**asdict(d), "paragraphs": sum(p.doc_id == doc_id for p in e.paras)}
