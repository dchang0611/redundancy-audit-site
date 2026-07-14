from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
impact_path = ROOT / "trained_hr_model_change_impact.json"
board_path = ROOT / "site" / "data" / "board.json"
output_path = ROOT / "site" / "data" / "model-impact-test.json"

impact = json.loads(impact_path.read_text(encoding="utf-8"))
board = json.loads(board_path.read_text(encoding="utf-8"))
impact["current_saved_board"] = board.get("backtest", {}).get("summary", [])
output_path.write_text(json.dumps(impact, indent=2), encoding="utf-8")
print(f"Saved: {output_path}")
