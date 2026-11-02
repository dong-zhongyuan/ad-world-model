import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_WORK = Path(os.environ.get("AD_FIGURE_WORK", str(REPO_ROOT / "figure_work")))
