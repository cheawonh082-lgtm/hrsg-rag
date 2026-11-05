"""Cloudflare 배포 — 배포판 앱(Render)이 쓰는 논문 검색 Worker (Workers AI bge-m3 + Vectorize).

    # 0) .env에 CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN (README "Cloudflare 배포")
    # 1) 로컬 BGE-M3 ↔ Workers AI 벡터 일치 확인 — 로컬로 만든 코퍼스 벡터를 그대로 올려도 되는지
    python scripts/cloudflare_sync.py parity --sample 20
    # 2) 코퍼스 벡터 + 문단 본문·서지 정보(메타데이터)를 Vectorize에 업로드 (인덱스 자동 생성)
    python scripts/cloudflare_sync.py upload                 # 로컬 인덱스 벡터 (parity 통과 시)
    python scripts/cloudflare_sync.py upload --reembed       # Workers AI로 다시 임베딩해 업로드 (parity 실패 시)
    # 3) Worker 배포 (용어 사전·코퍼스 규모 내보내기 → wrangler deploy → 비밀키 등록)
    python scripts/cloudflare_sync.py deploy
    # 4) 확인
    python scripts/cloudflare_sync.py query "감온기 분사수 제어 방식"

서지 정보(제목 등)를 고친 뒤에는 upload만 다시 하면 된다(벡터는 같고 메타데이터만 바뀐다).
자격 증명: 환경변수 CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN (.env 파일은 Git에서 제외)
RAG_API_KEY: 앱 ↔ Worker 비밀키. 없으면 deploy가 만들어 .env에 저장한다 — 같은 값을 앱(Render) 환경변수에도 넣는다.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.cloudflare_vectorize import (  # noqa: E402
    Credentials,
    VectorizeClient,
    WorkersAIEmbedder,
    check_parity,
    paragraph_metadata,
)
from rag.corpus_all import load_corpus_all  # noqa: E402
from rag.env import load_dotenv  # noqa: E402
from rag.glossary import KO_EN  # noqa: E402
from rag.rag_config import RagSettings  # noqa: E402
from rag.vector_index import LocalVectorIndex  # noqa: E402

WORKER_DIR = ROOT / "cloudflare" / "worker"
DEFAULT_INDEX = "hrsg-bge-m3"
SAMPLE_QUERIES = ["감온기 분사수 제어 방식", "HRSG 핀치 포인트 온도차의 의미", "superheater tube fouling effect on exhaust back pressure"]


def write_worker_data(corpus: Path, out_dir: Path = WORKER_DIR / "src") -> dict:
    """Worker에 함께 묶는 용어 사전(glossary.json)과 코퍼스 규모(stats.json)"""
    docs, paras = load_corpus_all(corpus)
    meta_path = RagSettings().bge_index / "meta.json"
    version = json.loads(meta_path.read_text(encoding="utf-8")).get("corpus_version") if meta_path.exists() else None
    stats = {"documents": len(docs), "paragraphs": len(paras), "corpus_version": version}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "glossary.json").write_text(json.dumps(KO_EN, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    return stats


def upload(creds: Credentials, corpus: Path, index_dir: Path, name: str, reembed: bool) -> None:
    docs, paras = load_corpus_all(corpus)
    index = LocalVectorIndex.load(index_dir)
    if index.paragraph_ids != [p.id for p in paras]:
        raise SystemExit(f"{index_dir}가 {corpus}와 맞지 않습니다 — scripts/build_rag_index.py로 인덱스를 다시 만드세요")
    metadata = [paragraph_metadata(docs[p.doc_id], p) for p in paras]
    if reembed:
        print(f"Workers AI로 {len(paras)}개 문단 임베딩 중...")
        vectors = WorkersAIEmbedder(creds).embed([p.embed_input for p in paras])
    else:
        vectors = index.vectors
    client = VectorizeClient(creds)
    client.ensure_index(name, dimensions=vectors.shape[1], metric="cosine")
    mutations = client.upsert(name, index.paragraph_ids, vectors, metadata)
    print(f"{len(paras)}개 업서트 요청 완료 (mutation {len(mutations)}건, 반영은 비동기 — 1~2분 뒤 query로 확인)")


def _save_env(key: str, value: str, path: Path = ROOT / ".env") -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = [ln for ln in lines if not ln.strip().startswith(f"{key}=")] + [f"{key}={value}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def deploy(corpus: Path) -> None:
    stats = write_worker_data(corpus)
    print(f"Worker 데이터: 문헌 {stats['documents']} · 문단 {stats['paragraphs']}")
    key = os.environ.get("RAG_API_KEY")
    if not key:
        key = secrets.token_urlsafe(32)
        _save_env("RAG_API_KEY", key)
        print("RAG_API_KEY를 새로 만들어 .env에 저장했습니다 — 같은 값을 앱(Render)의 RAG_API_KEY에 넣으세요")
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        raise SystemExit("npx를 찾을 수 없습니다 — Node.js를 설치하세요")
    env = {**os.environ}  # CLOUDFLARE_ACCOUNT_ID · CLOUDFLARE_API_TOKEN을 wrangler가 읽는다
    subprocess.run([npx, "--yes", "wrangler@4", "deploy"], cwd=WORKER_DIR, env=env, check=True)
    subprocess.run([npx, "--yes", "wrangler@4", "secret", "put", "RAG_API_KEY"], cwd=WORKER_DIR, env=env,
                   input=key, text=True, check=True)
    print("배포 완료 — 위에 나온 https://hrsg-rag-search.<서브도메인>.workers.dev 주소를 앱의 RAG_API_URL에 넣으세요")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=RagSettings().corpus)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.99)
    u = sub.add_parser("upload")
    u.add_argument("--index", type=Path, default=RagSettings().bge_index)
    u.add_argument("--name", default=DEFAULT_INDEX)
    u.add_argument("--reembed", action="store_true", help="로컬 벡터 대신 Workers AI로 다시 임베딩")
    sub.add_parser("worker-data")
    sub.add_parser("deploy")
    q = sub.add_parser("query")
    q.add_argument("--name", default=DEFAULT_INDEX)
    q.add_argument("--top-k", type=int, default=10)
    q.add_argument("text")
    args = ap.parse_args(argv)

    if args.cmd == "worker-data":
        print(write_worker_data(args.corpus))
        return
    if args.cmd == "deploy":
        deploy(args.corpus)
        return
    creds = Credentials.from_env()
    if args.cmd == "parity":
        from rag.embedder import BGEM3Embedder  # 로컬 모델(torch)은 parity에서만 필요

        _, paras = load_corpus_all(args.corpus)
        # 긴 문단도 표본에 넣는다 — Workers AI가 긴 입력을 다르게 자를 수 있다
        sample = paras[: args.sample // 2] + sorted(paras, key=lambda p: -len(p.text))[: args.sample - args.sample // 2]
        texts = [p.embed_input for p in sample] + SAMPLE_QUERIES
        report = check_parity(BGEM3Embedder(), WorkersAIEmbedder(creds), texts)
        print(f"{len(texts)}개 문장 — 코사인 최소 {report.min:.4f}, 평균 {report.mean:.4f}")
        print("✓ 로컬 벡터를 그대로 올려도 됩니다 (upload)" if report.ok(args.threshold)
              else "✗ 차이가 큽니다 — upload --reembed 로 Workers AI 벡터를 올리세요")
    elif args.cmd == "upload":
        upload(creds, args.corpus, args.index, args.name, args.reembed)
    else:
        vector = WorkersAIEmbedder(creds).embed([args.text])[0]
        _, paras = load_corpus_all(args.corpus)
        by_id = {p.id: p for p in paras}
        for pid, score in VectorizeClient(creds).query(args.name, vector, args.top_k):
            p = by_id.get(pid)
            print(f"{score:.4f}  {pid}  {p.text[:80] if p else ''}")


if __name__ == "__main__":
    main()
