import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.bm25 import BM25Searcher, glossary_expander, tokenize
from rag.cloudflare_vectorize import CloudflareError, _vectors_from
from rag.corpus import FieldMap, Paragraph, load_corpus
from rag.embedder import HashingEmbedder
from rag.evaluation import EvalQuestion, aggregate, load_questions, score_one
from rag.hybrid_search import HybridSearcher, SearchHit, diversify, rrf_fuse
from rag.vector_index import LocalVectorIndex

PARAS = [
    Paragraph("p1", "A", "Attemperator spray water controls superheater outlet temperature."),
    Paragraph("p2", "A", "Desuperheater nozzles inject feedwater into the steam line."),
    Paragraph("p3", "A", "Attemperator spray flow increases at low load."),
    Paragraph("p4", "B", "IAPWS-IF97 defines industrial formulations for steam properties."),
    Paragraph("p5", "C", "Pinch point temperature difference sets evaporator surface area."),
]


def q(qid, gold, lang="en", split="dev"):
    return EvalQuestion(qid, "?", lang, split, frozenset(gold))


# ---- RRF: 협업안 6쪽 예시 ----
def test_rrf_matches_proposal_example():
    fused = dict(rrf_fuse([["x", "y"], ["z", "w", "x"]], k=60))
    assert fused["x"] == pytest.approx(1 / 61 + 1 / 63)
    assert round(fused["x"], 4) == 0.0323
    assert fused["y"] == pytest.approx(1 / 62)
    assert max(fused, key=fused.get) == "x"  # 두 검색 모두 상위인 문단이 1위


def test_diversify_limits_two_per_doc():
    hits = [SearchHit(f"p{i}", "A" if i < 4 else f"D{i}", 1 - i / 10) for i in range(12)]
    out = diversify(hits, per_doc=2, final_k=8)
    assert [h.paragraph_id for h in out[:3]] == ["p0", "p1", "p4"]
    assert sum(h.doc_id == "A" for h in out) == 2 and len(out) == 8


# ---- 지표: 협업안 7쪽 계산 예시 (recall 0.50, MRR 0.42) ----
def test_metrics_match_proposal_example():
    rows = [
        (q("Q1", ["A", "B"]), *score_one([("A", None), ("X", None)], q("Q1", ["A", "B"]))),
        (q("Q2", ["C"]), *score_one([("X", None), ("Y", None), ("Z", None), ("C", None)], q("Q2", ["C"]))),
        (q("Q3", ["D"]), *score_one([("X", None)], q("Q3", ["D"]))),
    ]
    s = aggregate(rows)[("all", "all")]
    assert s.recall == pytest.approx(0.50)
    assert round(s.mrr, 2) == 0.42
    assert s.zero_hit == 1


def test_recall_counts_each_gold_doc_once():
    # 같은 정답 문헌의 문단이 두 개 나와도 정답 수는 1
    assert score_one([("A", None), ("A", None)], q("q", ["A", "B"])) == (0.5, 1.0)


def test_page_unit_requires_matching_page():
    question = EvalQuestion("q", "?", "en", "dev", frozenset({"A"}), frozenset({("A", 3)}))
    assert score_one([("A", 2), ("A", 3)], question, unit="page") == (1.0, 0.5)


# ---- BM25 ----
def test_tokenizer_keeps_standard_codes():
    assert {"iapws-if97", "iapws", "if97"} <= set(tokenize("IAPWS-IF97 table")) and "b31a" in tokenize("ASME B31A")


def test_bm25_exact_term_and_glossary():
    bm25 = BM25Searcher(PARAS)
    assert bm25.search("IF97", 3)[0].paragraph_id == "p4"
    assert bm25.search("IAPWS-IF97 steam", 1)[0].paragraph_id == "p4"
    assert bm25.search("감온기", 3) == []  # 용어 목록이 없으면 한국어로 영어 문단을 못 찾는다 (협업안 3절 한계)
    expanded = BM25Searcher(PARAS, query_expander=glossary_expander({"감온기": ["attemperator", "desuperheater"]}))
    assert {h.doc_id for h in expanded.search("감온기", 3)} == {"A"}


# ---- 하이브리드 ----
def test_hybrid_end_to_end_with_test_embedder(tmp_path):
    emb = HashingEmbedder(128)
    index = LocalVectorIndex.build(PARAS, emb.embed([p.text for p in PARAS]), {"model": emb.name})
    index.save(tmp_path / "idx")
    loaded = LocalVectorIndex.load(tmp_path / "idx")
    assert loaded.paragraph_ids == [p.paragraph_id for p in PARAS] and loaded.meta["dim"] == 128

    hybrid = HybridSearcher(loaded, emb, BM25Searcher(PARAS), top_n=5, final_k=8)
    hits = hybrid.search("attemperator spray water")
    assert hits[0].doc_id == "A"
    assert sum(h.doc_id == "A" for h in hits) <= 2  # 문헌당 최대 2문단
    assert hits[0].vector_rank is not None and hits[0].bm25_rank is not None


# ---- 로더 자동 감지 ----
def test_corpus_loader_autodetects_fields(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"paragraphs": [{"chunk_id": 1, "ref_id": "R1", "content": "x", "page": 4}]}), encoding="utf-8")
    (p,) = load_corpus(path)
    assert (p.paragraph_id, p.doc_id, p.text, p.page) == ("1", "R1", "x", 4)
    with pytest.raises(KeyError, match="사용 가능한 키"):
        load_corpus(path, FieldMap(text="body"))


def test_question_loader_handles_unanswerable_and_languages(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([
        {"id": "k1", "question": "감온기?", "lang": "ko", "split": "test", "evidence": [{"doc_id": "A", "page": [3, 4]}]},
        {"id": "e1", "query": "none?", "language": "English", "answerable": False},
    ]), encoding="utf-8")
    k1, e1 = load_questions(path)
    assert (k1.lang, k1.split, k1.gold_docs, k1.gold_pages) == ("ko", "test", frozenset({"A"}), frozenset({("A", 3), ("A", 4)}))
    assert (e1.lang, e1.split, e1.answerable) == ("en", "dev", False)


# ---- Cloudflare ----
def test_workers_ai_response_shapes():
    assert _vectors_from({"response": [[0.1, 0.2]], "shape": [1, 2]}) == [[0.1, 0.2]]
    assert _vectors_from({"data": [[0.3]], "shape": [1, 1]}) == [[0.3]]
    with pytest.raises(CloudflareError):
        _vectors_from({"request_id": "x"})


@pytest.mark.skipif(
    not (os.environ.get("CLOUDFLARE_ACCOUNT_ID") and os.environ.get("CLOUDFLARE_API_TOKEN")),
    reason="Cloudflare 자격 증명이 없으면 건너뜀",
)
def test_local_and_workers_ai_vectors_match():
    pytest.importorskip("sentence_transformers")
    from rag.cloudflare_vectorize import Credentials, WorkersAIEmbedder, check_parity
    from rag.embedder import BGEM3Embedder

    texts = ["감온기 분사수 제어", "Attemperator spray water controls superheater outlet temperature."]
    report = check_parity(BGEM3Embedder(), WorkersAIEmbedder(Credentials.from_env()), texts)
    assert report.ok(0.99), report.cosines


def test_normalized_vectors_give_cosine():
    v = HashingEmbedder(16).embed(["a b", "a b"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1) and float(v[0] @ v[1]) == pytest.approx(1)


def test_language_balanced_search_includes_both_languages():
    from rag.corpus import language_groups

    paras = PARAS + [Paragraph("k1", "K", "감온기는 과열 증기 온도를 조절한다"), Paragraph("k2", "K2", "드럼 수위 제어와 팽창 현상")]
    emb = HashingEmbedder(128)
    index = LocalVectorIndex.build(paras, emb.embed([p.text for p in paras]))
    groups = language_groups(paras)
    assert set(groups["ko"]) == {"k1", "k2"}
    hybrid = HybridSearcher(index, emb, None, top_n=3, lang_groups=groups)
    docs = {h.doc_id for h in hybrid.search_vector_only("attemperator spray water")}
    assert docs & {"K", "K2"} and docs & {"A", "B", "C"}  # 영어 질문에도 한국어 문헌 후보가 들어온다
