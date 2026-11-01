"""Cloudflare 시험 스크립트 (오프라인 평가가 끝난 뒤 실행).

    # 1) 로컬 BGE-M3 ↔ Workers AI 벡터 일치 확인 (코퍼스 앞부분 문단 + 질문 예시)
    python scripts/cloudflare_sync.py parity --corpus data/corpus.json --sample 20
    # 2) 로컬 인덱스를 Vectorize에 업로드 (1,024차원 · cosine 인덱스 자동 생성)
    python scripts/cloudflare_sync.py upload --index index/bge-m3 --name hrsg-bge-m3
    # 3) Workers AI 질문 벡터로 Vectorize 조회
    python scripts/cloudflare_sync.py query --name hrsg-bge-m3 "감온기 분사수 제어 방식"

자격 증명: 환경변수 CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN (.env 파일은 Git에서 제외)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.cloudflare_vectorize import Credentials, VectorizeClient, WorkersAIEmbedder, check_parity  # noqa: E402
from rag.corpus import FieldMap, load_corpus  # noqa: E402
from rag.embedder import BGEM3Embedder  # noqa: E402
from rag.env import load_dotenv  # noqa: E402
from rag.vector_index import LocalVectorIndex  # noqa: E402

SAMPLE_QUERIES = ["감온기 분사수 제어 방식", "HRSG 핀치 포인트 온도차의 의미", "superheater tube fouling effect on exhaust back pressure"]


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--corpus", default="data/corpus.json")
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.99)
    u = sub.add_parser("upload")
    u.add_argument("--index", default="index/bge-m3")
    u.add_argument("--name", default="hrsg-bge-m3")
    q = sub.add_parser("query")
    q.add_argument("--name", default="hrsg-bge-m3")
    q.add_argument("--top-k", type=int, default=40)
    q.add_argument("text")
    args = ap.parse_args(argv)

    creds = Credentials.from_env()
    if args.cmd == "parity":
        texts = [p.text for p in load_corpus(args.corpus, FieldMap())[: args.sample]] + SAMPLE_QUERIES
        report = check_parity(BGEM3Embedder(), WorkersAIEmbedder(creds), texts)
        print(f"{len(texts)}개 문장 — 코사인 최소 {report.min:.4f}, 평균 {report.mean:.4f}")
        print("✓ 섞어 써도 됩니다" if report.ok(args.threshold) else "✗ 차이가 큽니다 — 코퍼스를 Workers AI로 다시 임베딩하세요")
    elif args.cmd == "upload":
        index = LocalVectorIndex.load(args.index)
        client = VectorizeClient(creds)
        client.ensure_index(args.name, dimensions=index.vectors.shape[1], metric="cosine")
        mutations = client.upsert(args.name, index.paragraph_ids, index.vectors, [{"doc_id": d} for d in index.doc_ids])
        print(f"{len(index.paragraph_ids)}개 업서트 요청 완료 (mutation {len(mutations)}건, 반영은 비동기)")
    else:
        vector = WorkersAIEmbedder(creds).embed([args.text])[0]
        for pid, score in VectorizeClient(creds).query(args.name, vector, args.top_k):
            print(f"{score:.4f}  {pid}")


if __name__ == "__main__":
    main()
