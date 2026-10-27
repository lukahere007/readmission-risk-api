import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def encounter() -> dict:
    return json.loads((ROOT / "examples" / "single.json").read_text())
