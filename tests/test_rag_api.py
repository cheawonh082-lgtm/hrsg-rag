"""RAG API — 코퍼스 병합, 토크나이저, 언어 판별, 가중 RRF, 1~4안, HTTP 엔드포인트 (모델 없이 HashingEmbedder로)."""

from __future__ import annotations

import json

import pytest

from rag.corpus_all import load_corpus_all, merge_sources
from rag.embedder import HashingEmbedder
from rag.lang import detect_lang
from rag.rag_config import RagSettings, parse_weights
from rag.rag_engine import RagEngine, weighted_rrf
from rag.tokenizer import tokenize
from rag.vector_index import LocalVectorIndex

DOCS = {
    "R001": ("ko", [("1. 서론", "배열회수보일러(HRSG)의 핀치 포인트를 낮추면 증기 생산량이 늘어난다.", ["핀치 포인트", "HRSG"])]),
    "R002": ("en", [("2. Method", "Steam properties follow IAPWS-IF97 with enthalpy in kJ/kg.", ["IAPWS-IF97"]),
                    ("3. Results", "The HP/IP/LP drums operate at 120, 30 and 5 bar.", ["HP/IP/LP"]),
                    ("4. Capture", "Amine scrubbing removes 90 % of CO2 from the flue gas.", ["CO2 capture"])]),
}


def _write_sources(folder):
    folder.mkdir()
    mapping = []
    for doc_id, (lang, paras) in DOCS.items():
        doc = {"doc_id": doc_id, "title": f"Title {doc_id}", "authors": [], "year": 2020, "venue": "", "doi": "",
               "url": "", "license": "", "source_type": "paraphrase", "lang": lang}
        rows = [{"id": f"{doc_id}-{i:03d}", "doc_id": doc_id, "section": s, "page_start": 1, "page_end": 1, "text": t,
                 "terms": terms, "content_type": "text", "source_type": "paraphrase"}
                for i, (s, t, terms) in enumerate(paras, start=1)]
        (folder / f"{doc_id}.json").write_text(json.dumps({"document": doc, "paragraphs": rows, "report": {}}), encoding="utf-8")
        mapping.append({"doc_id": doc_id, "source_file": f"{doc_id}.pdf", "paragraph_count": len(rows)})
    (folder / "doc_id_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")


@pytest.fixture()
def settings(tmp_path):
    _write_sources(tmp_path / "src")
    data, warnings = merge_sources(tmp_path / "src")
    assert not warnings
    s = RagSettings(corpus=tmp_path / "corpus_all.json", index_dir=tmp_path / "index", cache_dir=tmp_path / "cache")
    s.corpus.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    _, paras = load_corpus_all(s.corpus)
    emb = HashingEmbedder(64)
    for path in (s.bge_index, s.qwen_index):
        LocalVectorIndex(emb.embed([p.embed_input for p in paras]), [p.id for p in paras], [p.doc_id for p in paras],
                         {"model": "hashing"}).save(path)
    return s


@pytest.fixture()
def engine(settings):
    return RagEngine(settings, embedders={"bge": HashingEmbedder(64), "qwen": HashingEmbedder(64)})


def test_merge_validates_and_keeps_mapping(settings):
    docs, paras = load_corpus_all(settings.corpus)
    assert len(docs) == 2 and len(paras) == 4
    assert docs["R001"].source_file == "R001.pdf"
    assert paras[0].embed_input.startswith("[1. 서론] 배열회수보일러")
    assert "핀치 포인트" in paras[0].bm25_input  # section + text + terms
    assert paras[0].lang == "ko" and paras[1].lang == "en"


def test_merge_rejects_duplicate_ids(tmp_path):
    _write_sources(tmp_path / "src")
    f = tmp_path / "src" / "R002.json"
    raw = json.loads(f.read_text(encoding="utf-8"))
    raw["paragraphs"][1]["id"] = raw["paragraphs"][0]["id"]
    f.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="중복"):
        merge_sources(tmp_path / "src")


def test_tokenizer_protects_codes_units_and_uses_morphemes():
    toks = tokenize("배열회수보일러(HRSG)의 HP/IP/LP 증기, kJ/kg, IF97, CO₂, IAPWS-IF97, 600 ℃")
    for t in ("hrsg", "hp/ip/lp", "hp", "lp", "kj/kg", "if97", "co2", "iapws-if97", "°c", "보일러", "증기"):
        assert t in toks, t
    assert "의" not in toks and "if" not in toks and "co" not in toks


def test_detect_lang():
    assert detect_lang("HRSG 핀치 포인트란?") == "ko"
    assert detect_lang("What is the HRSG pinch point?") == "en"
    assert detect_lang("IF97") == "en"


def test_weighted_rrf_matches_plain_rrf_and_applies_weights():
    rankings = {"bge": ["a", "b"], "bm25": ["b", "c"]}
    plain = dict(weighted_rrf(rankings, 60))
    assert plain["b"] == pytest.approx(1 / 62 + 1 / 61)
    weighted = dict(weighted_rrf(rankings, 60, {"bge": 1.0, "bm25": 0.5}))
    assert weighted["c"] == pytest.approx(0.5 / 62)
    assert parse_weights("bge=1, qwen=1,bm25=0.5") == {"bge": 1.0, "qwen": 1.0, "bm25": 0.5}


def test_methods_use_their_retrievers(engine):
    for method, used in {"1": {"bge", "bm25"}, "2": {"qwen", "bm25"}, "3": {"bge", "qwen", "bm25"}}.items():
        out = engine.search("IF97 enthalpy kJ/kg", method, top_n=2)
        assert len(out["results"]) == 2
        assert set(out["weights"]) == used
        assert all(set(r["retrievers"]) <= used for r in out["results"])
    top = engine.search("IF97 enthalpy kJ/kg", "1")["results"][0]
    assert top["id"] == "R002-001" and top["retrievers"]["bm25"]["rank"] == 1


def test_method4_weights_follow_query_language(engine):
    ko = engine.search("핀치 포인트를 낮추면 어떻게 되나?", "4")
    en = engine.search("What happens to CO2 capture?", "4")
    assert ko["query_lang"] == "ko" and ko["weights"]["bm25"] == 0.0
    assert en["query_lang"] == "en" and en["weights"]["bm25"] == 1.0


def test_glossary_expands_korean_queries_only(engine):
    from rag.glossary import english_terms

    # 띄어쓰기 무시, 긴 용어 우선 ("고압 증발기"가 "증발기"를 덮는다)
    assert english_terms("고압증발기 튜브의 침착물") == ["HP evaporator", "tube", "deposit deposition"]
    ko = engine.search("배열 회수 보일러 핀치 포인트", "4")
    assert ko["search_query"].startswith("배열 회수 보일러 핀치 포인트 ") and "pinch point" in ko["search_query"]
    assert engine.search("What is a pinch point?", "4")["search_query"] == "What is a pinch point?"
    engine.settings.expand_ko = False
    assert engine.search("핀치 포인트", "4")["search_query"] == "핀치 포인트"


def test_missing_index_disables_methods(settings):
    import shutil
    shutil.rmtree(settings.qwen_index)
    e = RagEngine(settings, embedders={"bge": HashingEmbedder(64)})
    assert e.available_methods() == {"1": True, "2": False, "3": False, "4": False}
    out = e.compare("HP/IP/LP drum pressure")
    assert "results" in out["methods"]["1"] and "error" in out["methods"]["2"]


def test_http_endpoints(engine):
    from fastapi.testclient import TestClient

    from api import app as api

    api.set_engine(engine)
    try:
        with TestClient(api.app) as client:
            assert client.get("/health").json()["methods"]["4"] is True
            r = client.post("/search", json={"query": "HP/IP/LP drums", "method": "3", "top_n": 3}).json()
            assert r["method"] == "3" and len(r["results"]) == 3 and r["results"][0]["id"] == "R002-002"
            assert client.post("/search", json={"query": "x", "method": "9"}).status_code == 422
            assert set(client.post("/search/compare", json={"query": "CO2"}).json()["methods"]) == {"1", "2", "3", "4"}
            assert client.get("/paragraphs/R001-001").json()["title"] == "Title R001"
            assert client.get("/documents/R002").json()["paragraphs"] == 3
            assert client.get("/paragraphs/none").status_code == 404
    finally:
        api.set_engine(None)
