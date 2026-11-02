"""한↔영 전문용어 사전 — 한국어 질문에 영어 용어를 덧붙여 영어 문단과 맞붙게 한다.

코퍼스 문단의 95%가 영어라 한국어 질문은 대부분 교차언어 검색이 된다. BM25는 한국어 단어로 영어 문단을
찾을 수 없으므로, 질문 속 한국어 용어를 찾아 영어 용어를 붙인 질의(expand_query)를 쓴다.
사전은 HRSG·복합화력 일반 용어로 만들었다 (특정 평가 질문의 정답 문단에 맞춘 것이 아니다).
키는 띄어쓰기를 무시하고 비교하며, 긴 용어를 먼저 찾는다 ("고압 증발기"가 "증발기"보다 먼저).
"""

from __future__ import annotations

import re
from functools import lru_cache

KO_EN: dict[str, str] = {
    # 설비·계통
    "배열회수보일러": "HRSG heat recovery steam generator",
    "폐열회수보일러": "HRSG heat recovery steam generator waste heat boiler",
    "폐열보일러": "waste heat boiler HRSG",
    "복합화력": "combined cycle power plant CCGT CCPP",
    "복합발전": "combined cycle power plant CCGT",
    "복합사이클": "combined cycle",
    "열병합": "cogeneration CHP combined heat and power",
    "가스터빈": "gas turbine",
    "증기터빈": "steam turbine",
    "고압 터빈": "HP turbine high pressure",
    "중압 터빈": "IP turbine intermediate pressure",
    "저압 터빈": "LP turbine low pressure",
    "발전기": "generator",
    "복수기": "condenser",
    "냉각탑": "cooling tower",
    "급수": "feedwater",
    "급수펌프": "feedwater pump",
    "탈기기": "deaerator",
    "드럼": "drum",
    "드럼 수위": "drum level",
    "절탄기": "economizer",
    "증발기": "evaporator",
    "과열기": "superheater",
    "재열기": "reheater",
    "고압 절탄기": "HP economizer",
    "고압 증발기": "HP evaporator",
    "고압 과열기": "HP superheater",
    "중압 증발기": "IP evaporator",
    "저압 증발기": "LP evaporator",
    "저압 과열기": "LP superheater",
    "열교환기": "heat exchanger",
    "튜브": "tube",
    "핀 튜브": "finned tube",
    "핀 달린 튜브": "finned tube fin",
    "튜브 다발": "tube bundle bank",
    "덕트 버너": "duct burner",
    "덕트": "duct",
    "버너": "burner",
    "보조연소": "supplementary firing duct burner afterburning",
    "후연소": "afterburning supplementary firing",
    "완전 연소형": "fully fired",
    "연소기": "combustor",
    "굴뚝": "stack",
    "바이패스": "bypass",
    "댐퍼": "damper",
    "밸브": "valve",
    "감온기": "attemperator desuperheater spray",
    "스프레이": "spray attemperator",
    "입구 안내깃": "inlet guide vane IGV",
    "압축기": "compressor",
    "히트펌프": "heat pump",
    "지역난방": "district heating",
    "관류형": "once-through",
    "관류": "once-through",
    "자연순환": "natural circulation",
    "강제순환": "forced circulation",
    "삼중압": "triple pressure three-pressure",
    "삼압력": "triple pressure three-pressure",
    "이중압": "dual pressure double pressure",
    "단일압": "single pressure",
    "재열": "reheat",
    # 열역학·성능
    "핀치 포인트": "pinch point",
    "핀치포인트": "pinch point",
    "접근 온도": "approach temperature",
    "최소 온도차": "minimum temperature difference pinch",
    "온도차": "temperature difference",
    "엑서지": "exergy",
    "엑서지 파괴": "exergy destruction",
    "엑서지 손실": "exergy loss",
    "엑서지 효율": "exergy efficiency",
    "비가역성": "irreversibility",
    "엔탈피": "enthalpy",
    "엔트로피": "entropy",
    "열효율": "thermal efficiency",
    "효율": "efficiency",
    "출력": "power output",
    "순출력": "net power output",
    "열소비율": "heat rate",
    "부분부하": "part load",
    "부분 부하": "part load",
    "저부하": "low load",
    "고부하": "high load",
    "부하": "load",
    "탈설계": "off-design",
    "설계점": "design point",
    "외기 온도": "ambient temperature",
    "대기 온도": "ambient temperature",
    "배기가스": "exhaust gas flue gas",
    "배가스": "flue gas exhaust gas",
    "연도가스": "flue gas",
    "배기 온도": "exhaust temperature",
    "증기": "steam",
    "주증기": "main steam",
    "증기 유량": "steam flow",
    "유량": "flow rate mass flow",
    "질량유량": "mass flow rate",
    "압력강하": "pressure drop",
    "압력 강하": "pressure drop",
    "압력손실": "pressure loss drop",
    "배압": "back pressure",
    "압력비": "pressure ratio",
    "열전달": "heat transfer",
    "열전달 계수": "heat transfer coefficient",
    "복사": "radiation radiative",
    "대류": "convection",
    "열유속": "heat flux",
    "열손실": "heat loss",
    "물성": "properties",
    "비체적": "specific volume",
    "정압비열": "isobaric heat capacity specific heat",
    "불확도": "uncertainty",
    "상관식": "correlation",
    "민감도": "sensitivity",
    "최적화": "optimization",
    "열경제": "thermoeconomic",
    "발전 원가": "cost of electricity",
    "발전원가": "cost of electricity",
    "경제성": "economic",
    "열화": "degradation",
    "성능 시험": "performance test",
    "열성능 시험": "thermal performance test",
    "성능 감시": "performance monitoring",
    # 운전·제어·동특성
    "기동": "startup start-up",
    "정지": "shutdown",
    "운전 모드": "operating mode",
    "동적 모델": "dynamic model",
    "동특성": "dynamic behaviour transient",
    "과도": "transient",
    "제어": "control",
    "계단응답": "step response",
    "팽창 현상": "swell",
    "수위": "level",
    "유동 역전": "flow reversal",
    "유동": "flow",
    "병렬 채널": "parallel channels",
    "단상": "single-phase",
    "모델": "model",
    "시뮬레이션": "simulation",
    "시뮬레이터": "simulator",
    "공정 시뮬레이터": "process simulator",
    "오픈소스": "open source",
    # 재료·화학·손상
    "부식": "corrosion",
    "유동가속부식": "flow-accelerated corrosion FAC",
    "침착물": "deposit deposition",
    "스케일": "scale deposit",
    "파울링": "fouling",
    "수처리": "water treatment chemistry",
    "산화성 전휘발성 처리": "AVT(O) oxidizing all-volatile treatment",
    "전휘발성": "all-volatile treatment AVT",
    "환원제": "reducing agent oxygen scavenger",
    "구리 합금": "copper alloy",
    "파손": "failure",
    "튜브 파손": "tube failure",
    "크리프": "creep",
    "피로": "fatigue",
    "수명": "lifetime life",
    "두꺼운 부재": "thick wall component",
    "누설": "leak leakage",
    # 배출·포집·연료
    "질소산화물": "NOx nitrogen oxides",
    "저질소산화물": "low NOx",
    "건식 저질소산화물": "dry low NOx DLN",
    "이산화탄소": "CO2 carbon dioxide",
    "이산화황": "SO2 sulfur dioxide",
    "탈황": "desulfurization FGD",
    "석회석": "limestone",
    "포집": "capture",
    "연소 후 포집": "post-combustion capture",
    "흡수제": "solvent absorbent",
    "아민": "amine MEA",
    "흡수": "absorption",
    "재생": "regeneration",
    "재생기": "regenerator stripper reboiler",
    "탈수": "dehydration",
    "압축": "compression",
    "천연가스": "natural gas",
    "합성가스": "syngas",
    "메탄": "methane",
    "수소": "hydrogen",
    "연료": "fuel",
    "미분탄": "pulverized coal",
    "석탄": "coal",
    "가스화기": "gasifier",
    "산소": "oxygen",
    "노즐": "nozzle",
    "입자": "particle",
    "전기로": "electric furnace",
    "페로실리콘": "ferrosilicon",
    "디젤": "diesel",
    "유기 랭킨": "organic Rankine cycle ORC",
    "랭킨": "Rankine",
    "리파워링": "repowering",
    "원자력": "nuclear",
    "발전소": "power plant",
}


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


@lru_cache(maxsize=1)
def _entries() -> list[tuple[str, str]]:
    """(띄어쓰기 없앤 키, 영어) — 긴 키부터"""
    return sorted(((_squash(k), v) for k, v in KO_EN.items()), key=lambda kv: -len(kv[0]))


def english_terms(query: str) -> list[str]:
    """질문에 나온 한국어 용어의 영어 표현. 긴 용어가 먼저 잡히고, 잡힌 부분은 짧은 용어가 다시 쓰지 않는다."""
    text = _squash(query)
    taken = [False] * len(text)
    found: list[tuple[int, str]] = []
    for key, en in _entries():
        start = text.find(key)
        while start != -1:
            end = start + len(key)
            if not any(taken[start:end]):
                taken[start:end] = [True] * len(key)
                found.append((start, en))
            start = text.find(key, end)
    return list(dict.fromkeys(en for _, en in sorted(found)))


def expand_query(query: str) -> str:
    """한국어 질문 + 영어 용어. 사전에 걸리는 용어가 없으면 질문 그대로."""
    terms = english_terms(query)
    return f"{query} {' '.join(terms)}" if terms else query
