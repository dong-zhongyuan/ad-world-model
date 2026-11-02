import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRUG_WORK = Path(os.environ.get("AD_DRUG_WORK", str(REPO_ROOT / "work" / "drug_discovery")))
