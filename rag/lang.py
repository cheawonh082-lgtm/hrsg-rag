"""언어 판별 — 한글 비율.

비율 = 한글 음절 수 / (한글 음절 + 라틴 문자 수). 숫자·기호는 세지 않는다.
"HRSG 핀치 포인트란?"처럼 영어 약어가 섞인 한국어 질문도 ko로 판별되도록 기준을 0.3으로 둔다.
"""

from __future__ import annotations

import re

_HANGUL = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
_LATIN = re.compile(r"[A-Za-z]")
KO_RATIO = 0.3


def hangul_ratio(text: str) -> float:
    h = len(_HANGUL.findall(text))
    total = h + len(_LATIN.findall(text))
    return h / total if total else 0.0


def detect_lang(text: str, threshold: float = KO_RATIO) -> str:
    """'ko' | 'en'. 글자가 없으면(숫자·기호만) 'en'."""
    return "ko" if hangul_ratio(text) >= threshold else "en"
