"""Protein-to-AD MR, adapted from the copied run_brain_mr.R formulas."""
import math
import numpy as np
from scipy.special import logsumexp
BASES=set("ACGT")
def finite(x):
    try:return math.isfinite(float(x))
    except (TypeError,ValueError):return False
def canonical_rsid(rsid,aliases=None):
    seen=set()
    while aliases and rsid in aliases:
        if rsid in seen:raise ValueError('Cyclic identifier aliases')
        seen.add(rsid);rsid=aliases[rsid]
    return rsid
def harmonize(e,y,frequency_tolerance=.10,aliases=None,allow_coordinate_identity=False):
    """Exact variant/build matching; exact indels; frequency-resolve palindromic SNPs."""
    if e['build']!=y['build']:return None,'build_mismatch'
    if str(e['chrom'])!=str(y['chrom']) or int(e['pos'])!=int(y['pos']):return None,'position_mismatch'
    a,b,c,d=[str(x).upper() for x in [e['effect_allele'],e['other_allele'],y['effect_allele'],y['other_allele']]]
    if any(not x or not set(x)<=BASES for x in [a,b,c,d]) or a==b or c==d:return None,'invalid_alleles'
    if canonical_rsid(e['rsid'],aliases)!=canonical_rsid(y['rsid'],aliases):
        if not allow_coordinate_identity or set([a,b])!=set([c,d]):return None,'rsid_mismatch'
    if any(len(x)!=1 for x in [a,b,c,d]):
        # Same rsID/build/position already checked. Accept exact sequence alleles only;
        # never infer strand, left-normalization, or padded indel equivalence.
        if (a,b)==(c,d):return 1,'indel_exact_aligned'
        if (a,b)==(d,c):return -1,'indel_exact_swapped'
        return None,'indel_requires_normalization'
    comp=lambda x:x.translate(str.maketrans('ACGT','TGCA'))
    if set([a,b]) in [set('AT'),set('CG')]:
        if set([a,b])!=set([c,d]):return None,'allele_mismatch'
        if not finite(e.get('eaf_exposure')) or not finite(y.get('eaf_outcome')):return None,'palindrome_missing_frequency'
        f,g=float(e['eaf_exposure']),float(y['eaf_outcome'])
        if not (0<f<1 and 0<g<1):return None,'palindrome_invalid_frequency'
        if min(f,1-f)>=.42 or min(g,1-g)>=.42:return None,'palindrome_ambiguous_frequency'
        direct=abs(f-g)<=frequency_tolerance
        flipped=abs(f-(1-g))<=frequency_tolerance
        if direct==flipped:return None,'palindrome_frequency_unresolved'
        return (1,'palindrome_frequency_aligned') if direct else (-1,'palindrome_frequency_flipped')
    if (a,b)==(c,d) or (comp(a),comp(b))==(c,d):return 1,'aligned'
    if (a,b)==(d,c) or (comp(a),comp(b))==(d,c):return -1,'swapped'
    return None,'allele_mismatch'
def wald(bx,sx,by,sy):
    bx,sx,by,sy=map(float,[bx,sx,by,sy])
    if not all(math.isfinite(x) for x in [bx,sx,by,sy]) or bx==0 or sx<=0 or sy<=0:raise ValueError('invalid_effect_or_SE')
    b=by/bx;s=sy/abs(bx);s2=math.sqrt(sy**2/bx**2+by**2*sx**2/bx**4)
    p=math.erfc(abs(b/s)/math.sqrt(2));p2=math.erfc(abs(b/s2)/math.sqrt(2))
    exp=lambda v:math.exp(v) if v<709 else math.inf
    return dict(beta_MR=b,se_MR=s,se_second_order=s2,p_MR=p,p_second_order=p2,
        CI_low=b-1.959963984540054*s,CI_high=b+1.959963984540054*s,
        OR=exp(b),OR_low=exp(b-1.959963984540054*s),OR_high=exp(b+1.959963984540054*s))
def bh(ps):
    p=np.asarray(ps,dtype=float)
    if np.any(~np.isfinite(p)) or np.any(p<0) or np.any(p>1):raise ValueError('invalid_p')
    if len(p)==0:return []
    order=np.argsort(p,kind='stable');values=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty(len(p));out[order]=np.minimum(values,1);return out.tolist()
def estimate_sdy(se,frequency,n):
    """coloc::sdY.est: no-intercept regression of 2*N*f*(1-f) on 1/var(beta)."""
    s,f=np.asarray(se,dtype=float),np.asarray(frequency,dtype=float)
    if len(s)!=len(f) or len(s)<2 or n<=0:raise ValueError('invalid_sdy_inputs')
    if np.any(~np.isfinite(s)) or np.any(s<=0) or np.any(~np.isfinite(f)) or np.any(f<=0) or np.any(f>=1):raise ValueError('invalid_sdy_statistics')
    x=1/(s*s);y=2*n*f*(1-f)
    return float(np.sqrt(np.dot(x,y)/np.dot(x,x)))
def coloc_abf(bx,sx,by,sy,prior_sd_x=.15,prior_sd_y=.2,p1=1e-4,p2=1e-4,p12=1e-5):
    """Single-causal-variant coloc ABF (Giambartolomei2014), beta/SE inputs.
    No outcome-P filtering is permitted when constructing the regional input.
    """
    x,sx,y,sy=[np.asarray(a,dtype=float) for a in [bx,sx,by,sy]]
    if not (len(x)==len(sx)==len(y)==len(sy)) or len(x)<2:raise ValueError('insufficient_or_mismatched_variants')
    if any(np.any(~np.isfinite(a)) for a in [x,sx,y,sy]) or np.any(sx<=0) or np.any(sy<=0):raise ValueError('invalid_regional_statistics')
    if not all(0<p<1 for p in [p1,p2,p12]) or min(prior_sd_x,prior_sd_y)<=0:raise ValueError('invalid_prior')
    def labf(b,s,sd):
        v=s*s;w=sd*sd
        return .5*(np.log(v/(v+w))+b*b*w/(v*(v+w)))
    lx,ly=labf(x,sx,prior_sd_x),labf(y,sy,prior_sd_y)
    ax,ay,axy=logsumexp(lx),logsumexp(ly),logsumexp(lx+ly)
    delta=axy-ax-ay
    cross=-np.inf if delta>=0 else ax+ay+np.log(-np.expm1(delta))
    logs=np.array([0,math.log(p1)+ax,math.log(p2)+ay,math.log(p1*p2)+cross,math.log(p12)+axy])
    probs=np.exp(logs-logsumexp(logs))
    return {f'PP_H{i}':float(v) for i,v in enumerate(probs)}
