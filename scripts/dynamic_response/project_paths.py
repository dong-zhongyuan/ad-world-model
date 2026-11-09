import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('AD_OMICS_ROOT', Path(__file__).resolve().parents[2])).resolve()
CONFIG = json.loads((ROOT/'configs/dynamic_response.json').read_text())
RESULTS = ROOT/'results/dynamic_response'


def configured_path(key):
    path = Path(CONFIG[key])
    return path if path.is_absolute() else ROOT/path
