"""루트 .env를 환경변수로 읽는다 (이미 설정된 값은 덮어쓰지 않음). .env는 Git에서 제외된다."""

import os
from pathlib import Path


def load_dotenv(path: str | Path = Path(__file__).resolve().parents[1] / ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
