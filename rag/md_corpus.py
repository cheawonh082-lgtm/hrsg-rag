"""EN/KO 병렬 md 코퍼스(rag_corpus/) → 청크 목록. 규칙은 rag_corpus/CORPUS_SPEC.md를 따른다.

- frontmatter(YAML)는 메타데이터로 분리하고 본문에 넣지 않는다 (title_ko 등이 영어 문서에 섞이지 않게).
- 본문은 `## N.` 절 단위로 자르고, 긴 절(영어본 기준 split_words 초과)은 `###` 단위로 나눈다.
  분할 여부는 논문(pair) 단위로 정해 EN/KO 청크 ID가 같은 자리끼리 대응하게 한다 (P07-en#s5.2 ↔ P07-ko#s5.2).
- `###`로 나눈 청크는 절의 첫 문장(논문 약칭)이 빠질 수 있어, 모든 청크 앞에 `제목 › 절 › 소제목` 경로를 붙인다.
- 논문 ID를 하드코딩하지 않고 폴더를 훑는다 — 논문이 늘어도 같은 명령으로 다시 만든다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REQUIRED_KEYS = (
    "doc_id", "paper_id", "pair_id", "lang", "parallel_doc_id", "title", "title_en", "title_ko",
    "authors", "year", "venue", "doi", "source_file", "source_coverage", "doc_type", "topics", "keywords",
)
TOPIC_VOCAB = frozenset("""
hrsg_design triple_pressure_reheat dual_pressure single_pressure energy_analysis exergy_analysis
thermodynamic_optimization part_load_operation operational_flexibility dynamic_simulation
fouling_slagging industrial_waste_heat cost_and_performance carbon_capture process_simulation_software
gas_turbine steam_turbine supplementary_firing ambient_conditions cycling_and_fatigue
plant_data_validation natural_gas_combined_cycle coal_power
""".split())
SECTIONS = 9
DEFAULT_SPLIT_WORDS = 700  # 스펙: 절이 ~700단어를 넘으면 ### 소제목으로 나눈다

# 코퍼스 구성(평가 모드) → 읽을 폴더. README의 '권장 측정 방식' 표와 같다.
VARIANTS = {
    "mixed": ("docs/en", "docs/ko"),        # 혼합·언어 균형
    "ko": ("docs/ko",),                     # 단일언어 (한글 질의 → 한국어 문서)
    "en_strict": ("docs_xling_strict/en",),  # 교차언어 핵심 (한글 질의 → 한글 0자 영어 문서)
}

_FRONT = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.S)
_H2 = re.compile(r"^## +(\d+)\.\s*(.+?)\s*$", re.M)
_H3 = re.compile(r"^### +(.+?)\s*$", re.M)


@dataclass
class Section:
    number: int
    heading: str
    intro: str                                   # 첫 ### 이전 본문 (없으면 절 전체)
    subs: list[tuple[str, str]] = field(default_factory=list)  # (소제목, 본문)

    @property
    def body(self) -> str:
        parts = [self.intro] + [f"### {h}\n{t}" for h, t in self.subs]
        return "\n\n".join(p for p in parts if p)


@dataclass
class MdDoc:
    path: Path
    meta: dict
    sections: list[Section]

    @property
    def doc_id(self) -> str:
        return self.meta["doc_id"]

    @property
    def title(self) -> str:
        return self.meta["title_ko"] if self.meta["lang"] == "ko" else self.meta["title_en"]


@dataclass(frozen=True)
class Chunk:
    paragraph_id: str   # 청크 ID — 기존 corpus.json 로더(load_corpus)와 호환되는 이름
    doc_id: str
    paper_id: str
    pair_id: str
    lang: str
    section: str        # "5" — 평가 질의의 gold_section과 비교
    sub: int            # 0 = 절 전체, 1.. = ### 순번
    heading: str
    text: str
    n_words: int


def words(text: str) -> int:
    return len(text.split())


def parse_md(path: Path) -> MdDoc:
    raw = path.read_text(encoding="utf-8")
    m = _FRONT.match(raw)
    if not m:
        raise ValueError(f"{path}: YAML frontmatter가 없습니다")
    meta = yaml.safe_load(m.group(1)) or {}
    body = raw[m.end():]
    heads = list(_H2.finditer(body))
    sections = []
    for i, h in enumerate(heads):
        text = body[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(body)].strip()
        subs_m = list(_H3.finditer(text))
        if subs_m:
            intro = text[: subs_m[0].start()].strip()
            subs = [(s.group(1), text[s.end(): subs_m[j + 1].start() if j + 1 < len(subs_m) else len(text)].strip())
                    for j, s in enumerate(subs_m)]
        else:
            intro, subs = text, []
        sections.append(Section(int(h.group(1)), h.group(2), intro, subs))
    return MdDoc(path, meta, sections)


def load_docs(root: Path, folders: tuple[str, ...]) -> list[MdDoc]:
    docs = [parse_md(p) for folder in folders for p in sorted((root / folder).glob("*.md"))]
    if not docs:
        raise FileNotFoundError(f"{root}의 {folders}에 md 문서가 없습니다")
    return docs


def validate(docs: list[MdDoc], expect_sections: int | None = SECTIONS,
             optional: frozenset[str] = frozenset()) -> tuple[list[str], list[str]]:
    """스펙 점검. (오류, 경고) — 오류가 있으면 코퍼스를 만들지 않는다. optional: 이 구성에서 빠져도 되는 키."""
    errors, warnings = [], []
    seen: dict[str, Path] = {}
    for d in docs:
        tag = d.path.name
        missing = [k for k in REQUIRED_KEYS if k not in d.meta and k not in optional]
        if missing:
            errors.append(f"{tag}: frontmatter 키 누락 {missing}")
            continue
        if d.doc_id in seen:
            errors.append(f"{tag}: doc_id {d.doc_id} 중복 ({seen[d.doc_id].name})")
        seen[d.doc_id] = d.path
        if d.doc_id != f"{d.meta['paper_id']}-{d.meta['lang']}":
            errors.append(f"{tag}: doc_id가 '{{paper_id}}-{{lang}}' 형식이 아닙니다")
        if not d.path.name.startswith(str(d.meta["paper_id"])):
            warnings.append(f"{tag}: 파일 이름이 paper_id로 시작하지 않습니다")
        nums = [s.number for s in d.sections]
        if expect_sections and nums != list(range(1, expect_sections + 1)):
            errors.append(f"{tag}: 절 번호 {nums} (1~{expect_sections} 필요)")
        bad_topics = [t for t in d.meta.get("topics") or [] if t not in TOPIC_VOCAB]
        if bad_topics:
            warnings.append(f"{tag}: 어휘표에 없는 topics {bad_topics}")
        for s in d.sections:
            if not s.body.strip():
                errors.append(f"{tag}: {s.number}절 본문이 비어 있습니다")
    # 병렬 쌍: 같은 pair_id끼리 절·소제목 구조가 같아야 청크가 대응한다
    pairs: dict[str, list[MdDoc]] = {}
    for d in docs:
        if "pair_id" in d.meta:
            pairs.setdefault(d.meta["pair_id"], []).append(d)
    for pid, group in pairs.items():
        shapes = {d.doc_id: [(s.number, len(s.subs)) for s in d.sections] for d in group}
        common = min(len(v) for v in shapes.values())
        trimmed = {k: v[:common] for k, v in shapes.items()}  # strict 변형은 9절이 없어 앞부분만 비교
        if len({json.dumps(v) for v in trimmed.values()}) > 1:
            errors.append(f"{pid}: EN/KO 절·소제목 구조가 다릅니다 {shapes}")
    return errors, warnings


def split_plan(docs: list[MdDoc], split_words: int) -> set[tuple[str, int]]:
    """(pair_id, 절 번호) 중 ### 단위로 나눌 절. 영어본 단어 수로 정하고 같은 pair의 한국어본에도 적용한다."""
    plan = set()
    for d in docs:
        if d.meta["lang"] != "en":
            continue
        for s in d.sections:
            if s.subs and words(s.body) > split_words:
                plan.add((d.meta["pair_id"], s.number))
    # 영어본이 없는 논문(한국어만 있는 경우): 한글은 띄어쓰기 단위가 길어 글자 수로 근사
    en_pairs = {d.meta["pair_id"] for d in docs if d.meta["lang"] == "en"}
    for d in docs:
        if d.meta["pair_id"] in en_pairs:
            continue
        for s in d.sections:
            if s.subs and len(s.body) > split_words * 5:
                plan.add((d.meta["pair_id"], s.number))
    return plan


def chunk_docs(docs: list[MdDoc], split: set[tuple[str, int]]) -> list[Chunk]:
    out = []
    for d in docs:
        m = d.meta
        for s in d.sections:
            path = f"{d.title} › {s.number}. {s.heading}"
            if (m["pair_id"], s.number) in split:
                pieces = [(i, f"{path} › {h}", (s.intro + "\n\n" if i == 1 and s.intro else "") + t)
                          for i, (h, t) in enumerate(s.subs, start=1)]
            else:
                pieces = [(0, path, s.body)]
            for sub, heading, body in pieces:
                cid = f"{d.doc_id}#s{s.number}" + (f".{sub}" if sub else "")
                out.append(Chunk(cid, d.doc_id, m["paper_id"], m["pair_id"], m["lang"], str(s.number), sub,
                                 heading, f"{heading}\n\n{body}", words(body)))
    return out


def doc_record(d: MdDoc, root: Path) -> dict:
    keep = ("doc_id", "paper_id", "pair_id", "lang", "parallel_doc_id", "title_en", "title_ko", "authors", "year",
            "venue", "doi", "source_coverage", "doc_type", "topics")
    return {k: d.meta.get(k) for k in keep} | {"path": d.path.relative_to(root).as_posix()}


def build(root: str | Path, variant: str, split_words: int = DEFAULT_SPLIT_WORDS) -> dict:
    """corpus.json 내용(dict). 오류가 있으면 ValueError."""
    root = Path(root)
    folders = VARIANTS[variant]
    docs = load_docs(root, folders)
    strict = variant == "en_strict"  # strict는 title_ko와 9절(용어표)을 뺀 판
    errors, warnings = validate(docs, SECTIONS - 1 if strict else SECTIONS, frozenset({"title_ko"}) if strict else frozenset())
    if errors:
        raise ValueError("코퍼스 스펙 오류:\n- " + "\n- ".join(errors))
    # 분할 기준은 항상 기본 영어본(docs/en)으로 정해 변형 간 청크 ID가 같게 한다
    ref_docs = docs if variant == "mixed" else docs + load_docs(root, ("docs/en",))
    chunks = chunk_docs(docs, split_plan(ref_docs, split_words))
    source_hash = hashlib.sha256(
        b"".join(d.path.read_bytes() for d in sorted(docs, key=lambda d: d.path.as_posix()))
    ).hexdigest()[:16]
    return {
        "meta": {"variant": variant, "folders": list(folders), "split_words": split_words,
                 "source_hash": source_hash, "n_docs": len(docs), "n_chunks": len(chunks), "warnings": warnings},
        "docs": [doc_record(d, root) for d in docs],
        "paragraphs": [asdict(c) for c in chunks],
    }
