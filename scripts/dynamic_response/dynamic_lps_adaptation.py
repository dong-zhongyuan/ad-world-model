"""Compatibility entry point for the current official v3 dynamic release."""
import os,runpy,sys
from pathlib import Path
root=Path(os.environ.get('AD_OMICS_ROOT', Path(__file__).resolve().parents[2]))
directory=root/'scripts/dynamic_response'
sys.path.insert(0,str(directory))
if __name__=='__main__':runpy.run_path(str(directory/'benchmark.py'),run_name='__main__')
