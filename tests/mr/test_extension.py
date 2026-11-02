import sys,unittest,tempfile,gzip,urllib.parse,csv,json,hashlib,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
if not (Path(__file__).resolve().parents[1]/'scripts/mr_core.py').exists():sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/mr'))
from mr_core import estimate_sdy,harmonize
class ExtensionTests(unittest.TestCase):
    def test_legacy_harmonised_coordinate_columns_and_effect_orientation(self):
        from mr_core import __file__ as core
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'data/mr/raw/ages').mkdir(parents=True);(r/'provenance/mr').mkdir(parents=True)
            raw=r/'data/mr/raw/ages/G1.tsv.gz'
            columns=['hm_rsid','hm_chrom','hm_pos','hm_other_allele','hm_effect_allele','hm_beta','hm_effect_allele_frequency','hm_code','variant_id','chromosome','base_pair_location','effect_allele','other_allele','beta','standard_error','p_value']
            with gzip.open(raw,'wt',newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(columns)
                w.writerow(['rs1',1,1000,'A','G',-.5,.8,10,'rs1',1,1000,'A','G',.5,.05,1e-12])
                w.writerow(['rs2',1,1100,'A','G',.001,.3,10,'rs2',1,1100,'G','A',.001,.05,.99])
                w.writerow(['rs3',1,2000000,'A','C',.001,.3,10,'rs3',1,2000000,'C','A',.001,.05,.99])
            x=dict(accession='GCST_TEST',gene='G1',SOMAmer='A1',filename='G1.tsv.gz',expected_md5=hashlib.md5(raw.read_bytes()).hexdigest(),
                   rsid='rs1',chrom='1',pos37=1001,effect_allele='A',beta_exposure=.5,se_exposure=.05,primary_within_gene=True,sample_size=5000,url='synthetic_test')
            (r/'provenance/mr/ages_download_plan.json').write_text(json.dumps([x]))
            (r/'provenance/mr/ages_variant_builds.json').write_text(json.dumps({'rs1':dict(pos37=1001,pos38=1000)}))
            subprocess.run([sys.executable,str(Path(core).with_name('extract_ages_regions.py')),'--root',str(r),'--assay','GCST_TEST'],check=True,capture_output=True)
            out=r/'data/mr/processed/ages/regions';e=json.loads((out/'GCST_TEST.instrument.json').read_text())
            self.assertEqual(e['pos'],1000);self.assertEqual(e['pos37'],1001);self.assertEqual(e['effect_allele'],'G');self.assertEqual(e['beta_exposure'],-.5)
            with gzip.open(out/'GCST_TEST.tsv.gz','rt') as f:rows=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual({x['rsid'] for x in rows},{'rs1','rs2'});self.assertEqual(float(rows[1]['p_exposure']),.99)
    def test_sdy_known_scale_and_rescaling(self):
        f=np.linspace(.01,.49,150);n=5300;s=2.3/np.sqrt(2*n*f*(1-f))
        self.assertAlmostEqual(estimate_sdy(s,f,n),2.3,places=12)
        self.assertAlmostEqual(estimate_sdy(s*4,f,n),9.2,places=12)
    def test_coordinate_identity_does_not_allow_allele_or_build_substitution(self):
        e=dict(rsid='rs1',build='GRCh38',chrom='1',pos=5,effect_allele='A',other_allele='G')
        y=dict(e,rsid='rs2',effect_allele='G',other_allele='A')
        self.assertIsNone(harmonize(e,y)[0]);self.assertEqual(harmonize(e,y,allow_coordinate_identity=True)[0],-1)
        y['other_allele']='T';self.assertIsNone(harmonize(e,y,allow_coordinate_identity=True)[0])
        y['other_allele']='A';y['build']='GRCh37';self.assertIsNone(harmonize(e,y,allow_coordinate_identity=True)[0])
    def test_bgzf_region_equals_full_scan(self):
        import pysam,range_tabix
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'test.tsv';lines=['#chrom\tpos\tref\talt\n']
            expected=[]
            for chrom in ['1','2']:
                for pos in range(1,120001,3):
                    lines.append(f'{chrom}\t{pos}\tA\tG\n')
                    if chrom=='1' and 16380<=pos<=65540:expected.append(pos)
            p.write_text(''.join(lines));gz=str(p)+'.gz';pysam.tabix_compress(str(p),gz,force=True)
            pysam.tabix_index(gz,seq_col=0,start_col=1,end_col=1,meta_char='#',force=True)
            original=range_tabix.get
            def local(url,start=None,end=None):
                b=Path(url).read_bytes();return b if start is None else b[start:end+1]
            try:
                range_tabix.get=local;t=range_tabix.RangeTabix(gz,Path(td)/'cache')
                rows=t.fetch('1',16380,65540)
                self.assertEqual([int(x['pos']) for x in rows],expected)
                self.assertEqual([int(x['pos']) for x in t.fetch('1',16384)],[16384])
            finally:range_tabix.get=original
if __name__=='__main__':unittest.main()
