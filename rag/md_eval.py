"""md 코퍼스(EN/KO 병렬) 오프라인 평가 — 질의 로더, 모드별 정답, 지표.

모드 (rag_corpus/README '권장 측정 방식')
- en_strict : 정답 = 영어 문서 (gold_by_lang.en, 없으면 {pair}-en). 한글 질의 → 영어 문서 = 교차언어
- ko        : 정답 = 한국어 문서 ({pair}-ko)
- mixed     : 정답 = pair 단위 (EN·KO 어느 쪽이 나와도 정답) + 상위 결과 중 질의 언어 문서 비율(언어 편향)

지표 (상위 final_k개 청크 기준, 정답이 하나뿐이므로 recall@k = hit@k)
- hit@1 / hit@3 / hit@8, MRR          : 정답 문서(또는 pair)의 첫 순위
- 절 hit@8 / 절 MRR                   : 정답 문서이면서 gold_section 절인 청크의 첫 순위 (근거 위치까지 맞는지)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

MODES = ("en_strict", "ko", "mixed")


@dataclass(frozen=True)
class MdQuery:
    qid: str
    query: str
    lang: str
    type: str
    subset: str          # base | ko_hangul_only | ...
    split: str           # dev | test
    pair_id: str
    gold_section: str
    gold_by_lang: dict = field(default_factory=dict, hash=False, compare=False)

    def gold_doc(self, mode: str) -> str | None:
        """모드별 정답 문서 ID. mixed는 pair 단위라 None."""
        if mode == "mixed":
            return None
        lang = "en" if mode == "en_strict" else "ko"
        return self.gold_by_lang.get(lang) or f"{self.pair_id}-{lang}"

    @property
    def group(self) -> str:
        """결과 표의 행: 한글 전용 / 기본-ko / 기본-en"""
        return "한글전용" if self.subset == "ko_hangul_only" else f"기본-{self.lang}"


def default_split(qid: str) -> str:
    """split 필드가 없을 때: qid 해시로 고정 배정(약 절반 dev). 논문이 추가돼도 기존 질의의 배정은 바뀌지 않는다."""
    return "dev" if int(hashlib.sha1(qid.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "test"


def load_md_queries(paths: Iterable[str | Path]) -> list[MdQuery]:
    out, seen = [], set()
    for path in paths:
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            r = json.loads(line)
            qid = str(r["qid"])
            if qid in seen:
                raise ValueError(f"{path}:{n} qid {qid} 중복")
            seen.add(qid)
            out.append(MdQuery(
                qid=qid, query=r["query"], lang=r["lang"], type=r.get("type", ""),
                subset=r.get("subset", "base"), split=r.get("split") or default_split(qid),
                pair_id=r.get("gold_pair_id") or r["gold_doc_ids"][0].rsplit("-", 1)[0],
                gold_section=str(r.get("gold_section", "")), gold_by_lang=r.get("gold_by_lang") or {},
            ))
    return out


@dataclass(frozen=True)
class ChunkInfo:
    doc_id: str
    pair_id: str
    section: str
    lang: str


@dataclass(frozen=True)
class QueryScore:
    rank: int | None          # 정답 문서(pair)의 첫 순위
    section_rank: int | None  # 정답 절의 첫 순위
    same_lang: float          # 상위 결과 중 질의 언어 청크 비율

    def hit(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k

    @property
    def rr(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0

    @property
    def section_rr(self) -> float:
        return 1.0 / self.section_rank if self.section_rank else 0.0


def score_query(ranked: Sequence[str], q: MdQuery, mode: str, info: dict[str, ChunkInfo]) -> QueryScore:
    """ranked: 순위대로 청크 ID (이미 final_k개로 잘린 목록)."""
    gold_doc = q.gold_doc(mode)

    def is_gold(c: ChunkInfo) -> bool:
        return c.pair_id == q.pair_id if gold_doc is None else c.doc_id == gold_doc

    rank = section_rank = None
    for r, cid in enumerate(ranked, start=1):
        c = info[cid]
        if is_gold(c):
            rank = rank or r
            if section_rank is None and c.section == q.gold_section:
                section_rank = r
    same = mean(info[c].lang == q.lang for c in ranked) if ranked else 0.0
    return QueryScore(rank, section_rank, same)


@dataclass
class Summary:
    scores: list[QueryScore] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.scores)

    def _avg(self, values) -> float:
        values = list(values)
        return mean(values) if values else float("nan")

    def hit(self, k: int) -> float:
        return self._avg(s.hit(k) for s in self.scores)

    @property
    def mrr(self) -> float:
        return self._avg(s.rr for s in self.scores)

    @property
    def section_hit8(self) -> float:
        return self._avg(s.section_rank is not None for s in self.scores)

    @property
    def section_mrr(self) -> float:
        return self._avg(s.section_rr for s in self.scores)

    @property
    def same_lang(self) -> float:
        return self._avg(s.same_lang for s in self.scores)

    @property
    def misses(self) -> int:
        return sum(s.rank is None for s in self.scores)

    def as_dict(self) -> dict:
        return {"n": self.n, "hit@1": self.hit(1), "hit@3": self.hit(3), "hit@8": self.hit(8), "mrr": self.mrr,
                "section_hit@8": self.section_hit8, "section_mrr": self.section_mrr,
                "same_lang_share": self.same_lang, "misses": self.misses}


def summarize(rows: Iterable[tuple[MdQuery, QueryScore]]) -> dict[str, Summary]:
    """행 키: all, dev, test, 그룹(한글전용/기본-ko/기본-en), 그룹·유형 (예: '한글전용·paraphrase')"""
    table: dict[str, Summary] = {}
    for q, s in rows:
        for key in ("all", q.split, q.group, f"{q.group}·{q.type}"):
            table.setdefault(key, Summary()).scores.append(s)
    return table
