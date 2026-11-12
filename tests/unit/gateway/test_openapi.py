"""Contract snapshot: any change to the public API shows up as a diff of `tests/unit/openapi.json`.

Regenerate after an intended change: `UPDATE_SNAPSHOTS=1 make test`.
"""

import json
import os
from pathlib import Path

from fastmri_knee.gateway.app import create_app
from fastmri_knee.gateway.settings import Settings

SNAPSHOT = Path(__file__).resolve().parents[1] / "openapi.json"


def test_openapi_matches_snapshot():
    spec = create_app(Settings(env="dev", auth_disabled=True)).openapi()
    text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_text(text)
    assert SNAPSHOT.exists(), "run with UPDATE_SNAPSHOTS=1 to create tests/unit/openapi.json"
    assert json.loads(SNAPSHOT.read_text()) == spec, "API changed: review the diff, then UPDATE_SNAPSHOTS=1"
