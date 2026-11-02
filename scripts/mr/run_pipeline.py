"""Reproducible CPU entry point; does not change models or training queues."""
from pathlib import Path
import argparse,os,subprocess,sys
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--stage',choices=['prepare','extract','mr','all'],default='all')
    a=ap.parse_args();here=Path(__file__).resolve().parent
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    steps={'prepare':['prepare_exposures.py','prepare_aliases.py'],'extract':['extract_outcome.py'],'mr':['run_mr.py'],
           'all':['prepare_exposures.py','prepare_aliases.py','extract_outcome.py','run_mr.py']}[a.stage]
    if a.stage in ['extract','all'] and not (a.root/'provenance/mr/ad_download_verified.json').exists():
        raise SystemExit('AD download/checksum pending. Run download_ad.py first; no missing association is imputed.')
    for step in steps:subprocess.run([sys.executable,str(here/step),'--root',str(a.root)],check=True,env=env)
if __name__=='__main__':main()
