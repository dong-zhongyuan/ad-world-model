import sys,unittest,math,tempfile,json,csv,subprocess,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
# Server tests live in tests/mr and scripts in scripts/mr.
if not (Path(__file__).resolve().parents[1]/'scripts/mr_core.py').exists():
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/mr'))
from mr_core import harmonize,wald,bh,coloc_abf
class MRTests(unittest.TestCase):
    def pair(self,ea='A',oa='G',ya='A',yo='G'):
        e=dict(build='GRCh38',chrom='1',pos=10,rsid='rs1',effect_allele=ea,other_allele=oa,eaf_exposure=.2)
        y=dict(build='GRCh38',chrom='1',pos=10,rsid='rs1',effect_allele=ya,other_allele=yo,eaf_outcome=.2)
        return e,y
    def test_alleles(self):
        for ea,oa,expected in [('A','G',1),('G','A',-1),('T','C',1),('C','T',-1)]:
            e,y=self.pair(ya=ea,yo=oa);self.assertEqual(harmonize(e,y)[0],expected)
    def test_build_and_identity(self):
        e,y=self.pair();y['build']='GRCh37';self.assertEqual(harmonize(e,y)[1],'build_mismatch')
        y['build']='GRCh38';y['rsid']='rs2';self.assertIsNone(harmonize(e,y)[0])
        y['rsid']='rs1';y['pos']=11;self.assertIsNone(harmonize(e,y)[0])
    def test_palindrome(self):
        e,y=self.pair('A','T','A','T');self.assertEqual(harmonize(e,y)[0],1)
        y['eaf_outcome']=.8;self.assertEqual(harmonize(e,y)[0],-1)
        y['eaf_outcome']=.5;self.assertIsNone(harmonize(e,y)[0])
        y['eaf_outcome']='';self.assertIsNone(harmonize(e,y)[0])
    def test_official_alias_requires_exact_variant(self):
        e,y=self.pair('A','AAT','A','AAT');y['rsid']='rs2';aliases={'rs1':'rs2'}
        self.assertIsNone(harmonize(e,y)[0])
        self.assertEqual(harmonize(e,y,aliases=aliases)[0],1)
        y['pos']=11;self.assertIsNone(harmonize(e,y,aliases=aliases)[0])
        y['pos']=10;y['other_allele']='AT';self.assertIsNone(harmonize(e,y,aliases=aliases)[0])
    def test_mismatch_indel(self):
        e,y=self.pair(ya='A',yo='C');self.assertIsNone(harmonize(e,y)[0])
        y['other_allele']='AG';self.assertIsNone(harmonize(e,y)[0])
    def test_wald_known(self):
        d=wald(.5,.05,.1,.02);self.assertAlmostEqual(d['beta_MR'],.2);self.assertAlmostEqual(d['se_MR'],.04)
        self.assertAlmostEqual(d['OR'],math.exp(.2));self.assertGreater(d['se_second_order'],d['se_MR'])
        self.assertAlmostEqual(d['p_MR'],5.733031437583866e-7,places=14)
    def test_indel_exact_only(self):
        e,y=self.pair('A','AAT','A','AAT');self.assertEqual(harmonize(e,y)[0],1)
        y['effect_allele']='AAT';y['other_allele']='A';self.assertEqual(harmonize(e,y)[0],-1)
        y['effect_allele']='AT';y['other_allele']='-';self.assertIsNone(harmonize(e,y)[0])
    def test_wald_sign(self):
        d=wald(-.5,.05,.1,.02);self.assertAlmostEqual(d['beta_MR'],-.2);self.assertGreater(d['se_MR'],0)
    def test_invalid_statistics(self):
        for vals in [(0,.1,.2,.1),(1,-.1,.2,.1),(1,.1,.2,0),(float('nan'),.1,.2,.1)]:
            with self.assertRaises(ValueError):wald(*vals)
    def test_global_bh(self):
        self.assertTrue(np.allclose(bh([.01,.04,.03,.8]),[.04,.053333333333,.053333333333,.8]))
        self.assertEqual(bh([]),[])
        with self.assertRaises(ValueError):bh([float('nan')])
    def test_coloc_enumerated_configurations(self):
        x=np.array([.1,.0,.2]);y=np.array([.12,.01,.18]);sx=np.array([.1]*3);sy=np.array([.1]*3)
        bf=lambda b,s,sd:np.sqrt(s*s/(s*s+sd*sd))*np.exp(b*b*sd*sd/(2*s*s*(s*s+sd*sd)))
        ax,ay=bf(x,sx,.15),bf(y,sy,.2)
        # Enumerate causal variant configurations independently of log-space implementation.
        weights=[1,1e-4*sum(ax),1e-4*sum(ay),1e-8*sum(ax[i]*ay[j] for i in range(3) for j in range(3) if i!=j),1e-5*sum(ax*ay)]
        expected=np.array(weights)/sum(weights);actual=coloc_abf(x,sx,y,sy)
        self.assertTrue(np.allclose(list(actual.values()),expected,atol=1e-12))
    def test_coloc_shared_vs_distinct(self):
        a=np.zeros(100);b=np.zeros(100);a[0]=1;b[0]=1
        shared=coloc_abf(a,np.ones(100)*.05,b,np.ones(100)*.05)
        b[0]=0;b[50]=1
        distinct=coloc_abf(a,np.ones(100)*.05,b,np.ones(100)*.05)
        self.assertGreater(shared['PP_H4'],.99);self.assertGreater(distinct['PP_H3'],.99)
    def test_runner_preserves_unavailable_candidates(self):
        from prepare_exposures import write_csv,sha
        from mr_core import __file__ as core
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)
            for p in ['data/mr/processed','provenance/mr','configs/mr','results/mr']:(r/p).mkdir(parents=True)
            ep=r/'data/mr/processed/primary_instruments.csv';yp=r/'data/mr/processed/ad_sentinel_statistics.csv'
            e,y=self.pair();e.update(gene='G1',assay='A1',beta_exposure=.5,se_exposure=.05,PAV_linked=False)
            y.update(rsid='rs_old_merged',beta_outcome=.1,se_outcome=.02,n_cases=100,n_controls=200)
            write_csv(ep,[e]);write_csv(yp,[y])
            write_csv(r/'data/mr/processed/candidate_coverage.csv',[dict(gene='G1',status='instrument_ready'),dict(gene='G2',status='not_in_ST3_assay_list')])
            (r/'configs/mr/protocol.json').write_text('{}')
            (r/'provenance/mr/exposure_lock.json').write_text(json.dumps(dict(instruments_sha256=sha(ep),candidate_version='test')))
            af=r/'provenance/mr/dbsnp_aliases.json';af.write_text(json.dumps(dict(instruments_sha256=sha(ep),aliases={'rs_old_merged':'rs1'})))
            (r/'provenance/mr/outcome_extraction.json').write_text(json.dumps(dict(status='complete',instruments_sha256=sha(ep),sentinels_sha256=sha(yp),protocol_sha256=sha(r/'configs/mr/protocol.json'),aliases_sha256=sha(af))))
            subprocess.run([sys.executable,str(Path(core).with_name('run_mr.py')),'--root',str(r)],check=True,capture_output=True)
            with (r/'results/mr/all_candidates.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2);self.assertEqual(rows[1]['mr_status'],'not_assessable_from_this_resource')
            self.assertEqual(rows[1]['p_MR'],'')
            self.assertEqual(rows[0]['mr_status'],'estimated')
    def test_extract_keeps_null_region_rows(self):
        import gzip
        from prepare_exposures import write_csv,sha
        from mr_core import __file__ as core
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)
            for p in ['data/mr/processed','data/mr/raw','provenance/mr','configs/mr']:(r/p).mkdir(parents=True)
            ep=r/'data/mr/processed/primary_instruments.csv'
            e,y=self.pair();e.update(gene='G1',assay='A1');write_csv(ep,[e])
            raw=r/'data/mr/raw/test.tsv.gz'
            cols=['variant_id','p_value','chromosome','base_pair_location','effect_allele','other_allele','effect_allele_frequency','beta','standard_error','n_cases','n_controls']
            with gzip.open(raw,'wt',newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(cols)
                w.writerows([['rs_new',.01,1,10,'A','G',.2,.1,.02,100,200],['rs2',.99,1,12,'C','T',.3,.0001,.02,90,180],['rs3',.8,1,1000,'A','C',.3,.001,.02,90,180]])
            md5=hashlib.md5(raw.read_bytes()).hexdigest()
            (r/'configs/mr/protocol.json').write_text(json.dumps({'outcome':{'md5':md5},'coloc':{'window_bp':100}}))
            (r/'provenance/mr/exposure_lock.json').write_text(json.dumps({'instruments_sha256':sha(ep)}))
            (r/'provenance/mr/dbsnp_aliases.json').write_text(json.dumps(dict(instruments_sha256=sha(ep),aliases={'rs1':'rs_new'})))
            (r/'provenance/mr/ad_download_verified.json').write_text(json.dumps({'status':'complete','path':str(raw),'bytes':raw.stat().st_size,'md5':md5}))
            subprocess.run([sys.executable,str(Path(core).with_name('extract_outcome.py')),'--root',str(r)],check=True,capture_output=True)
            with gzip.open(r/'data/mr/processed/ad_regions/G1.tsv.gz','rt') as f:rows=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual({x['rsid'] for x in rows},{'rs_new','rs2'})
            self.assertEqual(float(next(x for x in rows if x['rsid']=='rs2')['p_outcome']),.99)
            meta=json.loads((r/'provenance/mr/outcome_extraction.json').read_text());self.assertEqual(meta['gwas_rows_read'],3)
            with (r/'data/mr/processed/ad_sentinel_statistics.csv').open() as f:sentinels=list(csv.DictReader(f))
            self.assertEqual([x['rsid'] for x in sentinels],['rs_new'])
    def test_coloc_file_interface(self):
        import gzip
        from prepare_exposures import write_csv,sha
        from mr_core import __file__ as core
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)
            for p in ['configs/mr','data/mr/processed/ad_regions']:(r/p).mkdir(parents=True)
            request=dict(gene='G1',assay='A1',chrom='1',start=1,end=100)
            (r/'data/mr/processed/coloc_region_requests.json').write_text(json.dumps([request]))
            (r/'configs/mr/protocol.json').write_text(json.dumps({'coloc':{'minimum_common_variants':100,'p12':1e-5,'p12_sensitivity':[1e-6,1e-5,1e-4]}}))
            ex=[];ys=[]
            for i in range(100):
                e,y=self.pair();e.update(rsid='rs'+str(i),pos=i+1,beta_exposure=1 if i==0 else 0,se_exposure=.05)
                y.update(rsid='rs'+str(i),pos=i+1,beta_outcome=1 if i==0 else 0,se_outcome=.05)
                ex.append(e);ys.append(y)
            xp=r/'x.tsv'
            with xp.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(ex[0]),delimiter='\t');w.writeheader();w.writerows(ex)
            with gzip.open(r/'data/mr/processed/ad_regions/G1.tsv.gz','wt',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(ys[0]),delimiter='\t');w.writeheader();w.writerows(ys)
            meta=r/'meta.json';meta.write_text(json.dumps(dict(request,build='GRCh38',region_is_complete=True,filtered_by_p_value=False,exposure_sdY=1,data_sha256=sha(xp),sample_overlap='none')))
            subprocess.run([sys.executable,str(Path(core).with_name('run_coloc.py')),'--root',str(r),'--gene','G1','--pqtl-region',str(xp),'--metadata',str(meta)],check=True,capture_output=True)
            result=json.loads((r/'results/mr/coloc/G1.json').read_text())
            self.assertEqual(result['n_aligned'],100)
            self.assertGreater(next(x for x in result['results'] if x['primary'])['PP_H4'],.99)
if __name__=='__main__':unittest.main()
