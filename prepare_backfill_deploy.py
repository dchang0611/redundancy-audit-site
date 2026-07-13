from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

BASE = "https://dchang0611.github.io/redundancy-audit-site"
DATA = Path(__file__).resolve().parent / "site" / "data"
HISTORY = DATA / "history"


def download(url: str, destination: Path) -> None:
    try:
        with urlopen(url, timeout=20) as response:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(response.read())
    except Exception as exc:
        print(f"Skipped {url}: {exc}")


def main() -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)
    download(f"{BASE}/data/board.json", DATA / "board.json")
    download(f"{BASE}/data/latest-board.csv", DATA / "latest-board.csv")
    try:
        with urlopen(f"{BASE}/data/history/index.json", timeout=20) as response:
            prior_dates = json.load(response).get("dates", [])
    except Exception:
        prior_dates = []
    for slate_date in prior_dates:
        destination = HISTORY / f"{slate_date}.json"
        if not destination.exists():
            download(f"{BASE}/data/history/{slate_date}.json", destination)
    dates = sorted((p.stem for p in HISTORY.glob("????-??-??.json")), reverse=True)
    (HISTORY / "index.json").write_text(json.dumps({"dates": dates}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
