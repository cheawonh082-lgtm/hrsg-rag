"""BM25 토크나이저 — 한국어는 형태소 분석(kiwipiepy), 영어는 소문자화 후 토큰화.

규격·약어·단위가 잘게 쪼개지지 않게 먼저 '보호 토큰'으로 떼어 낸다 (Kiwi는 IF97 → IF+97, CO2 → CO+2로 나눈다).
- 영숫자 붙은 것은 한 토큰:        IF97 → if97, CO2 → co2, GT26 → gt26, 9FA → 9fa
- / - . · 로 이어진 것은 전체 + 부분: HP/IP/LP → hp/ip/lp, hp, ip, lp · kJ/kg → kj/kg, kj, kg
                                    IAPWS-IF97 → iapws-if97, iapws, if97 · 숫자만인 1.5는 부분을 만들지 않음
- 아래첨자·전각 문자는 NFKC로 정규화: CO₂ → co2, ℃ → °c
남은 한글은 Kiwi로 형태소를 나눠 내용어(명사·어근·용언 어간)만 남긴다. 조사·어미는 버린다.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_COMPOUND = re.compile(r"°[CFK]|[A-Za-z0-9]+(?:[-/.·][A-Za-z0-9]+)*")
_SEP = re.compile(r"[-/.·]")
_NUMERIC = re.compile(r"[0-9]+(?:[.,][0-9]+)*")
_HANGUL_RUN = re.compile(r"[가-힣]+(?:\s+[가-힣]+)*")
KEEP_TAGS = frozenset({"NNG", "NNP", "XR", "VV", "VA", "SH", "NR"})


@lru_cache(maxsize=1)
def _kiwi():
    try:
        from kiwipiepy import Kiwi
    except ImportError as e:
        raise ImportError("한국어 BM25에는 kiwipiepy가 필요합니다: pip install kiwipiepy") from e
    return Kiwi()


def _latin_tokens(text: str) -> list[str]:
    out = []
    for m in _COMPOUND.finditer(text):
        tok = m.group(0).lower()
        out.append(tok)
        if _SEP.search(tok) and not _NUMERIC.fullmatch(tok):
            out.extend(part for part in _SEP.split(tok) if part)
    return out


def _korean_tokens(text: str) -> list[str]:
    runs = " ".join(m.group(0) for m in _HANGUL_RUN.finditer(text))
    if not runs:
        return []
    return [t.form for t in _kiwi().tokenize(runs) if t.tag in KEEP_TAGS]


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    return _latin_tokens(text) + _korean_tokens(text)
