"""Cloudflare Workers AI(@cf/baai/bge-m3) + Vectorize 연동 (협업안 1절 ⑤ — 로컬 오프라인 평가 후 시험).

자격 증명은 환경변수로만 받는다 (코드·파일·커밋에 넣지 않음):
    CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN   (토큰 권한: Workers AI Read, Vectorize Edit)

주요 함수
- WorkersAIEmbedder.embed(texts): Workers AI로 임베딩 (로컬 Embedder와 같은 인터페이스, L2 정규화)
- check_parity(local, remote, texts): 같은 문장에 대한 로컬·Workers AI 벡터의 코사인 유사도 — 1에 가까워야
  로컬로 만든 코퍼스 벡터와 Workers AI 질문 벡터를 섞어 쓸 수 있다. 낮으면 코퍼스를 Workers AI로 다시 임베딩.
- VectorizeClient: 인덱스 생성(1,024차원, cosine) · NDJSON 업서트 · 질의

API 참고 (2026-10 확인)
- Workers AI: POST /accounts/{id}/ai/run/@cf/baai/bge-m3  body {"contexts": [{"text": ...}]} (query 없이 → 임베딩)
- Vectorize v2: POST /vectorize/v2/indexes · /indexes/{name}/upsert (application/x-ndjson, HTTP 1회 최대 5,000개)
  · /indexes/{name}/query  {"vector", "topK", "returnMetadata"}  — topK 최대 100(메타데이터·값 미반환 시), 50(반환 시)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import requests

from rag.embedder import DIM, Embedder, normalize

API = "https://api.cloudflare.com/client/v4"
WORKERS_AI_MODEL = "@cf/baai/bge-m3"
UPSERT_BATCH = 1000  # HTTP 한도 5,000보다 작게 — 요청 크기 제한 여유
EMBED_BATCH = 50


class CloudflareError(RuntimeError):
    pass


@dataclass(frozen=True)
class Credentials:
    account_id: str
    api_token: str

    @classmethod
    def from_env(cls) -> "Credentials":
        account, token = os.environ.get("CLOUDFLARE_ACCOUNT_ID"), os.environ.get("CLOUDFLARE_API_TOKEN")
        if not account or not token:
            raise CloudflareError("CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN 환경변수를 설정하세요 (.env 참고).")
        return cls(account, token)


class _Client:
    def __init__(self, creds: Credentials, timeout: float = 60.0):
        self._creds = creds
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {creds.api_token}"

    def _url(self, path: str) -> str:
        return f"{API}/accounts/{self._creds.account_id}{path}"

    def _call(self, method: str, path: str, *, retries: int = 3, **kwargs) -> dict:
        for attempt in range(retries):
            res = self._session.request(method, self._url(path), timeout=self._timeout, **kwargs)
            if res.status_code == 429 or res.status_code >= 500:
                time.sleep(2**attempt)
                continue
            body = res.json() if res.content else {}
            if not res.ok or not body.get("success", True):
                errors = body.get("errors") or res.text[:300]
                raise CloudflareError(f"{method} {path} → {res.status_code}: {errors}")
            return body.get("result", body)
        raise CloudflareError(f"{method} {path} → 재시도 {retries}회 실패 ({res.status_code})")


class WorkersAIEmbedder(_Client):
    name = WORKERS_AI_MODEL
    dim = DIM

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = list(texts[i : i + EMBED_BATCH])
            result = self._call("POST", f"/ai/run/{WORKERS_AI_MODEL}", json={"contexts": [{"text": t} for t in batch]})
            vectors = _vectors_from(result)
            if len(vectors) != len(batch):
                raise CloudflareError(f"응답 벡터 수 {len(vectors)} ≠ 요청 {len(batch)}")
            out.extend(vectors)
        arr = np.asarray(out, dtype=np.float32)
        if arr.shape[1] != self.dim:
            raise CloudflareError(f"차원 {arr.shape[1]} ≠ {self.dim}")
        return normalize(arr)


def _vectors_from(result: dict) -> list[list[float]]:
    """bge-m3 응답의 벡터 목록. 문서에 출력 스키마가 없어 알려진 형태(response / data)를 모두 받는다."""
    for key in ("response", "data"):
        value = result.get(key)
        if isinstance(value, list) and value and isinstance(value[0], list):
            return value
    raise CloudflareError(f"Workers AI 응답에서 임베딩을 찾지 못했습니다. 키: {sorted(result)}")


@dataclass(frozen=True)
class ParityReport:
    cosines: list[float]

    @property
    def min(self) -> float:
        return min(self.cosines)

    @property
    def mean(self) -> float:
        return float(np.mean(self.cosines))

    def ok(self, threshold: float = 0.99) -> bool:
        return self.min >= threshold


def check_parity(local: Embedder, remote: Embedder, texts: Sequence[str]) -> ParityReport:
    a, b = local.embed(texts), remote.embed(texts)
    return ParityReport([float(x) for x in np.sum(a * b, axis=1)])


class VectorizeClient(_Client):
    def create_index(self, name: str, dimensions: int = DIM, metric: str = "cosine", description: str = "") -> dict:
        body = {"name": name, "config": {"dimensions": dimensions, "metric": metric}, "description": description}
        return self._call("POST", "/vectorize/v2/indexes", json=body)

    def get_index(self, name: str) -> dict | None:
        try:
            return self._call("GET", f"/vectorize/v2/indexes/{name}", retries=1)
        except CloudflareError as e:
            if "404" in str(e):
                return None
            raise

    def ensure_index(self, name: str, dimensions: int = DIM, metric: str = "cosine") -> dict:
        info = self.get_index(name)
        if info is None:
            return self.create_index(name, dimensions, metric)
        cfg = info.get("config", {})
        if cfg.get("dimensions") != dimensions or cfg.get("metric") != metric:
            raise CloudflareError(f"기존 인덱스 '{name}' 설정 {cfg}이(가) 요구({dimensions}, {metric})와 다릅니다.")
        return info

    def upsert(self, name: str, ids: Sequence[str], vectors: np.ndarray, metadata: Sequence[dict] | None = None) -> list[str]:
        """NDJSON 업서트. 반환: mutationId 목록 (반영은 비동기 — 바로 조회하면 일부가 안 보일 수 있다)"""
        for pid in ids:
            if len(pid.encode()) > 64:
                raise CloudflareError(f"벡터 ID는 64바이트 이하여야 합니다: {pid!r}")
        mutations = []
        for i in range(0, len(ids), UPSERT_BATCH):
            lines = []
            for j in range(i, min(i + UPSERT_BATCH, len(ids))):
                row = {"id": ids[j], "values": [float(x) for x in vectors[j]]}
                if metadata:
                    row["metadata"] = metadata[j]
                lines.append(json.dumps(row, ensure_ascii=False))
            result = self._call(
                "POST",
                f"/vectorize/v2/indexes/{name}/upsert",
                data="\n".join(lines).encode("utf-8"),
                headers={"Content-Type": "application/x-ndjson"},
            )
            mutations.append(result.get("mutationId", ""))
        return mutations

    def query(self, name: str, vector: Sequence[float], top_k: int = 40, return_metadata: str = "none") -> list[tuple[str, float]]:
        """반환: [(문단 ID, 코사인 유사도)] — 메타데이터 없이 받아야 topK를 100까지 쓸 수 있다."""
        limit = 100 if return_metadata == "none" else 50
        if top_k > limit:
            raise CloudflareError(f"topK {top_k} > {limit} (returnMetadata={return_metadata})")
        body = {"vector": [float(x) for x in vector], "topK": top_k, "returnValues": False, "returnMetadata": return_metadata}
        result = self._call("POST", f"/vectorize/v2/indexes/{name}/query", json=body)
        return [(m["id"], float(m["score"])) for m in result.get("matches", [])]
