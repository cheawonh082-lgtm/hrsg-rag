"""md 코퍼스 평가(3단계) — 모드별 정답 판정, 지표, split 배정, pair 중복 제거."""

from __future__ import annotations

import json

import pytest

from rag.hybrid_search import SearchHit, diversify
from rag.md_eval import ChunkInfo, default_split, load_md_queries, score_query, summarize

INFO = {
    "P01-ko#s5": ChunkInfo("P01-ko", "P01", "5", "ko"),
    "P01-en#s5": ChunkInfo("P01-en", "P01", "5", "en"),
    "P01-en#s6": ChunkInfo("P01-en", "P01", "6", "en"),
    "P02-en#s5": ChunkInfo("P02-en", "P02", "5", "en"),
    "P02-ko#s5": ChunkInfo("P02-ko", "P02", "5", "ko"),
}


@pytest.fixture()
def queries(tmp_path):
    rows = [
        {"qid": "P01-k01", "lang": "ko", "subset": "ko_hangul_only", "type": "factual", "query": "질의",
         "gold_pair_id": "P01", "gold_doc_ids": ["P01-en", "P01-ko"], "gold_section": "6",
         "gold_by_lang": {"en": "P01-en", "ko": "P01-ko"}},
        {"qid": "P01-q01", "lang": "en", "type": "factual", "query": "query",
         "gold_pair_id": "P01", "gold_doc_ids": ["P01-en", "P01-ko"], "gold_section": "5", "split": "test"},
    ]
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return load_md_queries([path])


def test_loader_fields_and_groups(queries):
    k, q = queries
    assert k.group == "한글전용" and q.group == "기본-en"
    assert q.split == "test"  # 파일의 split이 우선
    assert k.split == default_split("P01-k01") and k.split in {"dev", "test"}
    assert k.gold_doc("en_strict") == "P01-en" and q.gold_doc("ko") == "P01-ko" and k.gold_doc("mixed") is None


def test_duplicate_qid_is_rejected(tmp_path):
    line = json.dumps({"qid": "a", "lang": "ko", "query": "x", "gold_doc_ids": ["P01-ko"]})
    (tmp_path / "a.jsonl").write_text(line, encoding="utf-8")
    (tmp_path / "b.jsonl").write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match="중복"):
        load_md_queries([tmp_path / "a.jsonl", tmp_path / "b.jsonl"])


def test_doc_and_section_ranks_per_mode(queries):
    k = queries[0]  # 한글 질의, 정답 절 6
    ranked = ["P02-en#s5", "P01-en#s5", "P01-en#s6"]
    s = score_query(ranked, k, "en_strict", INFO)
    assert (s.rank, s.section_rank) == (2, 3)
    assert s.hit(1) is False and s.hit(3) is True and s.rr == 0.5
    # ko 모드에서는 영어 문서는 정답이 아니다
    assert score_query(ranked, k, "ko", INFO).rank is None


def test_mixed_counts_pair_and_language_share(queries):
    k = queries[0]
    s = score_query(["P02-ko#s5", "P01-ko#s5", "P01-en#s6"], k, "mixed", INFO)
    assert s.rank == 2 and s.section_rank == 3
    assert s.same_lang == pytest.approx(2 / 3)  # 한국어 질의, 한국어 청크 2/3


def test_summary_rows(queries):
    k, q = queries
    table = summarize([(k, score_query(["P01-en#s6"], k, "en_strict", INFO)),
                       (q, score_query(["P02-en#s5"], q, "en_strict", INFO))])
    assert table["all"].n == 2 and table["all"].hit(1) == 0.5 and table["all"].misses == 1
    assert table["한글전용·factual"].section_hit8 == 1.0 and table["test"].n == 1


def test_pair_group_dedupe():
    hits = [SearchHit(c, INFO[c].doc_id, 1.0) for c in ("P01-ko#s5", "P01-en#s5", "P01-en#s6", "P02-en#s5")]
    by_doc = diversify(hits, per_doc=1, final_k=8)
    by_pair = diversify(hits, per_doc=1, final_k=8, doc_group={"P01-ko": "P01", "P01-en": "P01"})
    assert [h.paragraph_id for h in by_doc] == ["P01-ko#s5", "P01-en#s5", "P02-en#s5"]
    assert [h.paragraph_id for h in by_pair] == ["P01-ko#s5", "P02-en#s5"]
