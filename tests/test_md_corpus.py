"""md 코퍼스 적재(1단계)와 임베딩 캐시 테스트 — 실제 rag_corpus 없이 임시 폴더로 확인한다."""

from __future__ import annotations

import numpy as np
import pytest

from rag.corpus import load_corpus
from rag.embed_cache import EmbeddingCache, text_key
from rag.md_corpus import build


def _doc(paper: str, lang: str, sections: dict[int, str], title_ko: bool = True) -> str:
    other = "ko" if lang == "en" else "en"
    front = [
        "---", f"doc_id: {paper}-{lang}", f"paper_id: {paper}", f"pair_id: {paper}", f"lang: {lang}",
        f"parallel_doc_id: {paper}-{other}", 'title: "T"', 'title_en: "Title EN"',
        *(['title_ko: "제목"'] if title_ko else []),
        'authors: ["A"]', "year: 2020", 'venue: "V"', 'doi: ""', 'source_file: "x.pdf"', "source_coverage: full_text",
        "doc_type: journal_article", "topics: [hrsg_design]", 'keywords: ["k"]', "---", "",
    ]
    body = [f"## {n}. Heading {n}\n\n{text}\n" for n, text in sections.items()]
    return "\n".join(front) + "\n".join(body)


LONG = "### Sub A\nIn Kim (2020) " + "word " * 400 + "\n\n### Sub B\n" + "more " * 400


@pytest.fixture()
def corpus_root(tmp_path):
    secs = {n: f"In Kim (2020), section {n} text." for n in range(1, 10)}
    secs[5] = "Intro sentence.\n\n" + LONG
    for lang in ("en", "ko"):
        (tmp_path / "docs" / lang).mkdir(parents=True)
        (tmp_path / "docs" / lang / f"P01_kim.{lang}.md").write_text(_doc("P01", lang, secs), encoding="utf-8")
    strict = {n: t for n, t in secs.items() if n < 9}
    (tmp_path / "docs_xling_strict" / "en").mkdir(parents=True)
    (tmp_path / "docs_xling_strict" / "en" / "P01_kim.en.md").write_text(
        _doc("P01", "en", strict, title_ko=False), encoding="utf-8")
    return tmp_path


def test_mixed_chunks_are_parallel_and_split(corpus_root):
    data = build(corpus_root, "mixed")
    ids = [c["paragraph_id"] for c in data["paragraphs"]]
    # 9절 중 5절만 ### 2개로 나뉜다 → 문서당 10청크, EN/KO 같은 자리
    assert len(ids) == 20
    assert "P01-en#s5.1" in ids and "P01-ko#s5.2" in ids and "P01-en#s5" not in ids
    first = next(c for c in data["paragraphs"] if c["paragraph_id"] == "P01-en#s5.1")
    assert "Intro sentence." in first["text"]  # 첫 ### 앞 문장은 첫 소청크에 붙는다
    assert first["text"].startswith("Title EN › 5. Heading 5 › Sub A")
    assert first["section"] == "5" and first["lang"] == "en"


def test_ko_chunk_heading_uses_korean_title(corpus_root):
    data = build(corpus_root, "ko")
    assert all(c["text"].startswith("제목 ›") for c in data["paragraphs"])
    assert "P01-ko#s5.2" in {c["paragraph_id"] for c in data["paragraphs"]}  # 분할은 영어본 기준


def test_strict_allows_missing_title_ko_and_section9(corpus_root):
    data = build(corpus_root, "en_strict")
    assert {c["section"] for c in data["paragraphs"]} == {str(n) for n in range(1, 9)}


def test_output_loads_with_existing_loader(corpus_root, tmp_path):
    import json
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(build(corpus_root, "mixed"), ensure_ascii=False), encoding="utf-8")
    paragraphs = load_corpus(path)
    assert paragraphs[0].paragraph_id == "P01-en#s1" and paragraphs[0].doc_id == "P01-en"


def test_spec_errors_stop_the_build(corpus_root):
    ko = corpus_root / "docs" / "ko" / "P01_kim.ko.md"
    ko.write_text(ko.read_text(encoding="utf-8").replace("## 9. Heading 9", "## 10. Heading 9"), encoding="utf-8")
    with pytest.raises(ValueError, match="절 번호"):
        build(corpus_root, "mixed")


def test_pair_structure_mismatch_is_an_error(corpus_root):
    ko = corpus_root / "docs" / "ko" / "P01_kim.ko.md"
    ko.write_text(ko.read_text(encoding="utf-8").replace("### Sub B", "Sub B"), encoding="utf-8")
    with pytest.raises(ValueError, match="구조가 다릅니다"):
        build(corpus_root, "mixed")


def test_embedding_cache_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "c")
    keys = [text_key("m", 512, t) for t in ("a", "b")]
    cache.add(keys, np.eye(2, dtype=np.float32))
    cache.add(keys[:1], np.zeros((1, 2), dtype=np.float32))  # 이미 있는 키는 덮어쓰지 않는다
    cache.save()
    again = EmbeddingCache(tmp_path / "c")
    assert len(again) == 2
    assert np.allclose(again.get(keys[0]), [1, 0]) and again.get(text_key("m", 8192, "a")) is None
