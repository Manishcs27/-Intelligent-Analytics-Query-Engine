import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import app

cases = {
    "Total sales in India for March": 108.0,
    "Top 2 cities by profit": 2,
}
for q, expected in cases.items():
    out = app.answer(q)
    assert out["confidence_score"] > 0
    assert out["result"]
print("smoke tests passed")
