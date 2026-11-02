"""Adapted from copied audit_ukb_cis.py. Freeze AD exposure selection before outcomes."""
from pathlib import Path
import argparse,csv,hashlib,json,math,zipfile
import xml.etree.ElementTree as ET
from datetime import datetime,timezone
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def sheet(path,name,header):
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS):
                strings.append(''.join(t.text or '' for t in si.findall('.//s:t',NS)))
        book=ET.fromstring(z.read('xl/workbook.xml'))
        rels={x.attrib['Id']:x.attrib['Target'] for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        s=next(x for x in book.findall('s:sheets/s:sheet',NS) if x.attrib['name']==name)
        target=rels[s.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
        target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
        rows=[]
        for r in ET.fromstring(z.read(target)).findall('s:sheetData/s:row',NS):
            if int(r.attrib['r'])<header:continue
            cells={}
            for c in r.findall('s:c',NS):
                col=''.join(x for x in c.attrib['r'] if x.isalpha());n=0
                for x in col:n=n*26+ord(x.upper())-64
                v=c.find('s:v',NS);value=None if v is None else v.text
                if c.attrib.get('t')=='s' and value is not None:value=strings[int(value)]
                elif c.attrib.get('t')=='inlineStr':value=''.join(t.text or '' for t in c.findall('.//s:t',NS))
                elif value is not None:
                    try:value=float(value)
                    except ValueError:pass
                cells[n-1]=value
            if int(r.attrib['r'])==header:cols=[cells.get(i) for i in range(max(cells)+1)]
            else:rows.append({k:cells.get(i) for i,k in enumerate(cols) if k is not None})
        return rows
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_csv(path,rows,columns=None):
    columns=columns or list(rows[0])
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);r=ap.parse_args().root
    ref=r/'data/mr/reference';out=r/'data/mr/processed';out.mkdir(parents=True,exist_ok=True)
    workbook=ref/'41586_2023_6592_MOESM3_ESM.xlsx'
    edges=list(csv.DictReader((ref/'prescreened_cross_tissue_edges.csv').open()))
    hub=list(csv.DictReader((ref/'filtered_cross_tissue_edges.csv').open()))
    genes=sorted({x['source'] for x in edges});hub_genes={x['source'] for x in hub}
    assert len(edges)==2021 and len(genes)==136 and len(hub_genes)==138
    assays=sheet(workbook,'ST3',3);qtl=sheet(workbook,'ST9',5);pav=sheet(workbook,'ST14',3)
    by_assay={x['UKBPPP ProteinID']:x for x in assays if x.get('UKBPPP ProteinID')}
    pav_set={(x['UKBPPP ProteinID'],x['rsID (lead variant)']) for x in pav}
    leads=[]
    for x in qtl:
        assay=x.get('UKBPPP ProteinID');ann=by_assay.get(assay,{});gene=ann.get('Gene symbol')
        if gene not in genes or str(x.get('cis/trans')).lower()!='cis':continue
        vid=str(x['Variant ID (CHROM:GENPOS (hg37):A0:A1:imp:v1)']).split(':')
        bx,sx=float(x['BETA (discovery, wrt. A1)']),float(x['SE (discovery)'])
        F=(bx/sx)**2 if sx>0 else math.nan;ea,oa=vid[3].upper(),vid[2].upper()
        neglogp=float(x['log10(p) (discovery)'])
        sequence_alleles=bool(ea and oa) and set(ea)<=set('ACGT') and set(oa)<=set('ACGT') and ea!=oa
        eligible=math.isfinite(F) and F>=10 and bx!=0 and neglogp>=-math.log10(5e-8) and sequence_alleles
        leads.append(dict(assay=assay,gene=gene,uniprot=ann.get('UniProt'),chrom=str(int(x['CHROM'])),
            pos=int(x['GENPOS (hg38)']),build='GRCh38',pos37=int(vid[1]),rsid=x['rsID'],
            effect_allele=ea,other_allele=oa,beta_exposure=bx,se_exposure=sx,
            eaf_exposure=x['A1FREQ (discovery)'],neg_log10_p_exposure=neglogp,F_exposure=F,
            beta_replication=x.get('BETA (replication)'),se_replication=x.get('SE (replication)'),
            PAV_linked=(assay,x['rsID']) in pav_set,MHC=bool(x.get('MHC')),specimen='plasma',
            source='Sun2023_UKBPPP_ST9_discovery',exposure_unit='source_reported_protein_analysis_unit',
            n_exposure=34557,eligible_exposure=eligible,exclusion='' if eligible else 'invalid_sequence_or_weak_or_invalid_effect',
            primary=False,primary_rule='highest_F_then_assay_then_rsid_within_gene_before_outcomes'))
    for gene in genes:
        valid=sorted([x for x in leads if x['gene']==gene and x['eligible_exposure']],key=lambda x:(-x['F_exposure'],x['assay'],x['rsid']))
        if valid:valid[0]['primary']=True
    leads.sort(key=lambda x:(x['gene'],not x['primary'],-x['F_exposure']))
    coverage=[]
    for gene in genes:
        measured=[x for x in assays if x.get('Gene symbol')==gene];cis=[x for x in leads if x['gene']==gene]
        selected=[x for x in cis if x['primary']]
        coverage.append(dict(gene=gene,candidate_origin='v2_step3_actual_prescreened',
            n_edges=sum(x['source']==gene for x in edges),n_measured_assays=len(measured),n_published_cis=len(cis),
            n_eligible_cis=sum(x['eligible_exposure'] for x in cis),primary_assay=selected[0]['assay'] if selected else '',
            primary_rsid=selected[0]['rsid'] if selected else '',
            status='instrument_ready' if selected else ('published_cis_not_eligible' if cis else ('assayed_no_cis_in_ST9' if measured else 'not_in_ST3_assay_list')),
            ADT_required_for_MR=False))
    write_csv(out/'candidate_coverage.csv',coverage);write_csv(out/'all_published_cis_leads.csv',leads)
    write_csv(out/'primary_instruments.csv',[x for x in leads if x['primary']])
    write_csv(out/'candidate_hub_sensitivity.csv',[{'gene':g,'in_actual_136':g in genes} for g in sorted(hub_genes)])
    lock=dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate_version='v2_actual_step3_136',
        n_candidates=len(genes),n_edges=len(edges),n_primary=sum(x['primary'] for x in leads),
        source_sha256={x.name:sha(x) for x in [workbook,ref/'prescreened_cross_tissue_edges.csv',ref/'filtered_cross_tissue_edges.csv']},
        instruments_sha256=sha(out/'primary_instruments.csv'),selection_uses_outcome=False,
        primary_resource='Complete Sun2023 ST3/ST9/ST14; not PD-restricted table',
        source_url='https://www.nature.com/articles/s41586-023-06592-6',
        PAV_policy='Retain flagged primary; prespecified no-PAV sensitivity; may reflect binding rather than abundance',
        missing_policy='No eligible tool is unassessable, not negative',
        multi_instrument_policy='Strongest eligible cis lead per gene; do not pool others without LD',
        family='BH once across all eligible primary gene-to-AD estimates for fixed Sun resource',
        supplementary_resources='AGES is a separate independent-source extension, not result-driven fallback')
    target=r/'provenance/mr/exposure_lock.json'
    if target.exists():assert json.loads(target.read_text())['instruments_sha256']==lock['instruments_sha256'],'Frozen exposure set changed'
    else:target.write_text(json.dumps(lock,indent=2))
    print(json.dumps({'candidates':len(genes),'primary':lock['n_primary'],'cis_leads':len(leads),
        'status_counts':{s:sum(x['status']==s for x in coverage) for s in sorted({x['status'] for x in coverage})}},indent=2))
if __name__=='__main__':main()
