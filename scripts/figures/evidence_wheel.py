# User-supplied reference, preserved verbatim below the provenance line.
# fig5_e_wheel.py — Fig5b hero: 证据轮（cp AD figure3/src/panel_n_circular.py 改）
# 扇区 = 候选蛋白; 10 环 = 证据层（淡=未过 / 实=过）; 外环柱 = 过层数; 圆心 = 层图例
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pycirclize import Circos

# cp of the user-supplied wheel; data, sectors, labels, fonts and native size edited.
from pathlib import Path
import pandas as pd
import json
W = Path(__file__).resolve().parents[2] / 'figure_work'
(W/'build').mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':12,'pdf.fonttype':42,'savefig.bbox':None})
m = pd.read_csv(W/'data/evidence_gates_with_survival.csv')
TARGETS = list(m.protein.unique())
layers = ['network','mr_bell','mr_finn','direction_outcome','direction_platform','coloc_bell','coloc_finn','survival','drugclip','docking_qc']
names = ['Network top 50%','Bellenguez MR','FinnGen MR','Outcome direction','Platform direction','Bellenguez coloc','FinnGen coloc','ADNI MMSE association','DrugCLIP top 20','Docking pose QC']
colors = ['#599CAF','#B290C5','#9075AA','#CA7182','#AC5266','#68AC9D','#408C7C','#D49A63','#C6A35A','#6485A3']
all_cmap = [mcolors.ListedColormap([mcolors.to_rgba(c,.12),mcolors.to_rgba(c,1)]) for c in colors]
spec=json.loads((W/'page_layout.json').read_text());dim=spec['panels']['4b'];ww=dim['w']*.75;hh=dim['h']*.75
fig = plt.figure(figsize=(ww,hh))
ax = fig.add_axes([.04,.13,.92,.83],polar=True)
sector_group = {t:int(m.protein.eq(t).sum()) for t in TARGETS}
circo_group = Circos(sector_group, start=8, end=352, space=.2, endspace=False)
for sector in circo_group.sectors:
    z = m[m.protein.eq(sector.name)]
    start_radius = 78
    for i,(l,c) in enumerate(zip(layers,all_cmap)):
        radius = start_radius-i*3.15
        track_group=sector.add_track((radius-2.95,radius))
        track_group.heatmap(z[l].eq('pass').astype(int).to_numpy().reshape(1,-1),vmin=0,vmax=1,cmap=c,rect_kws=dict(ec='white',lw=.27))
        for j,status in enumerate(z[l]):
            if status=='unresolved':track_group.rect(j,j+1,fc='#F5F5F5',ec='#9EA8AE',lw=.2,hatch='//')
    x=np.arange(len(z))+.5
    Sum=z[layers].eq('pass').sum(axis=1)
    track_sum=sector.add_track((82,96))
    track_sum.bar(x,Sum.to_numpy(),vmin=0,vmax=len(layers),width=.76,color=['#B34F69' if v==len(layers) else '#597B85' for v in Sum],ec='none')
    track_sum.line(x,Sum.to_numpy(),vmin=0,vmax=len(layers),color='#526770',lw=.45,ls='--')
    if z.q_union.iloc[0] < .05:
        sector.text('NESP55' if sector.name=='GNAS' else sector.name,r=99,size=12,orientation='vertical',weight='bold' if z.hub50.iloc[0] else 'normal')
circo_group.plotfig(ax=ax)
# Ring key in the actual centre; labels fit without reducing the 12 pt font.
for i,(name,c) in enumerate(zip(names,colors)):
    yy=.654-i*.0275
    fig.patches.append(mpatches.Rectangle((.355,yy-.006),.016,.012,transform=fig.transFigure,color=c,clip_on=False))
    fig.text(.383,yy,name,va='center',ha='left',fontsize=12)
# Explicit visual encodings, not an additional screening threshold.
handles=[mpatches.Patch(facecolor='#6B8F85',label='Met'),mpatches.Patch(facecolor='#EDF1EF',edgecolor='#BAC5C1',label='Not met'),mpatches.Patch(facecolor='#F5F5F5',edgecolor='#9EA8AE',hatch='//',label='Unresolved'),mpatches.Patch(facecolor='#B34F69',label='All 10 met')]
fig.legend(handles=handles,loc='lower center',ncol=2,fontsize=12,frameon=False,bbox_to_anchor=(.5,.015),handlelength=1,handletextpad=.5,columnspacing=1.3)
fig.text(4/(ww*72),1-5/(hh*72),'B',fontsize=16,weight='bold',va='top')
fig.savefig(W/'build/4b.pdf')
# Register a native 4 x 4 module panel without resizing other panels.
p=W/'panel_manifest.json';manifest=json.loads(p.read_text()) if p.exists() else {};manifest['4b']=dict(id='4b',units_x=dim['w'],units_y=dim['h'],width_in=ww,height_in=hh,width_px=round(ww*600),height_px=round(hh*600),base_unit_px=450);p.write_text(json.dumps(manifest,indent=2))
p=W/'template_usage.json';u=json.loads(p.read_text()) if p.exists() else {};u['4b']=dict(template='user_evidence_wheel',source='full cp templates/expanded_periscope/evidence_wheel.py; binary tracks, count bars and central key retained',n=len(m),question='Which existing criteria does each primary assay meet? Outer bar equals the exact number passed.');p.write_text(json.dumps(u,indent=2))
plt.close(fig)
assert (m.passed_count == m[layers].eq('pass').sum(axis=1)).all()
print('DRAW 4b',len(m),'primary assays, 60 genes, 10 binary criteria')
