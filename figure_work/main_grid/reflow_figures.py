"""Reflow existing vector chart components; never use anisotropic scaling.
Figures 3/4 are authored separately from the frozen source tables in ggplot2.
All other plotted marks are taken losslessly from the current vector masters.
"""
from pathlib import Path
import json, string, shutil
import pymupdf as fitz
import numpy as np
P=Path(__file__).resolve().parent;O=P/'output';Q=P/'qa';S=P/'source'
layout=json.loads((S/'original_page_layout.json').read_text())
FONT=next(str(f) for f in [Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf'),Path('/public/home/mengxl/dzy/tools/fonts/Arial Bold.ttf')] if f.exists())
SCALE=180/25.4*72/1080
baseline=json.loads((S/'baseline_vector_layout.json').read_text())
records=[v for v in baseline if v['figure']==5]
spacing=[]
molecule_grid=[]
old_scale={(v['figure'],v['panel'],v['part']):v['scale_x'] for v in baseline if 'scale_x'in v}
# Common column grids are 0/540, 0/360/720 and 0/216/432/648/864.
plans={
1:[('a',0,0,1080,212),('b',0,216,540,460),('c',540,216,540,460),('d',0,680,540,250),('e',540,680,540,250),('legend',0,934,1080,43),('f',0,981,540,260),('g',540,981,540,260)],
2:[('a',0,0,1080,158),('b',0,162,720,350),('c',720,162,360,350),('d',0,516,360,318),('e',360,516,360,318),('f',720,516,360,318),('g',0,838,540,240),('h',540,838,540,240),('i',0,1082,540,300),('j',540,1082,540,300)],
5:[('a',0,0,1080,164),('b',0,168,540,526),('c',540,168,540,526),('d',0,698,540,270),('e',540,698,540,270),('f',0,972,360,256),('g',360,972,360,256),('h',720,972,360,256),('i',0,1232,540,230),('j',540,1232,540,230)],
6:[('a',0,0,1080,140),('b',0,144,1080,382),('c',0,530,360,230),('d',360,530,360,230),('e',720,530,360,230),('f',0,764,432,245),('g',432,764,648,245),('h',0,1013,1080,238)]+[(string.ascii_lowercase[8+i],(i%5)*216,1255+(i//5)*164,216,160) for i in range(10)]}

def tight(doc,clip):
    pix=doc[0].get_pixmap(clip=clip,matrix=fitz.Matrix(1.4,1.4),alpha=False)
    a=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
    ys,xs=np.where(a.min(axis=2)<245)
    if not len(xs):return clip
    return fitz.Rect(max(clip.x0,(pix.x+xs.min())/1.4-1),max(clip.y0,(pix.y+ys.min())/1.4-1),min(clip.x1,(pix.x+xs.max()+1)/1.4+1),min(clip.y1,(pix.y+ys.max()+1)/1.4+1))

def put(page,doc,clip,dest,n,l,part='chart',align='center'):
    src=tight(doc,fitz.Rect(clip));dest=fitz.Rect(dest)
    ratio=min(dest.width/src.width,dest.height/src.height)
    if part=='status legend':ratio=min(ratio,1.05)
    w,h=src.width*ratio,src.height*ratio
    x=dest.x0+(dest.width-w)/2;y=dest.y0 if align=='top' else dest.y0+(dest.height-h)/2
    actual=fitz.Rect(x,y,x+w,y+h)
    page.show_pdf_page(actual,doc,0,clip=src,keep_proportion=True)
    records.append(dict(figure=n,panel=l,part=part,source=list(src),destination=list(actual),scale_x=ratio,scale_y=ratio))

def place_fixed(page,doc,src,x,y,ratio,n,l,part):
    r=fitz.Rect(x,y,x+src.width*ratio,y+src.height*ratio)
    page.show_pdf_page(r,doc,0,clip=src,keep_proportion=True)
    records.append(dict(figure=n,panel=l,part=part,source=list(src),destination=list(r),scale_x=ratio,scale_y=ratio))
    return r

def balanced_columns(page,doc,clips,dest,n,l,ratio,label=None,legend=None,part='column'):
    dest=fitz.Rect(dest);dest.x0-=12;dest.x1+=12;body=doc
    if label:
        body=fitz.open(stream=doc.tobytes(),filetype='pdf')
        for block in body[0].get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    if span['text']==label:
                        r=fitz.Rect(span['bbox']);r.y0+=r.height*.3;r.y1-=r.height*.1
                        body[0].add_redact_annot(r,fill=(1,1,1))
        body[0].apply_redactions(images=0,graphics=0)
    srcs=[tight(body,fitz.Rect(c))for c in clips]
    top=min(r.y0 for r in srcs);bottom=max(r.y1 for r in srcs)
    srcs=[fitz.Rect(r.x0,top,r.x1,bottom)for r in srcs]
    leg=tight(doc,fitz.Rect(legend))if legend else None
    footer=(18 if label else 0)+(leg.height if leg else 0)
    gaps=3*(int(bool(label))+int(bool(legend)))
    ratio=min(ratio,dest.width/sum(r.width for r in srcs),(dest.height-gaps)/(bottom-top+footer))
    widths=[r.width*ratio for r in srcs];gap=(dest.width-sum(widths))/(len(srcs)+1)
    assert gap>=-1e-6
    cursor=dest.x0+gap;placed=[]
    for j,r in enumerate(srcs):
        placed.append(place_fixed(page,body,r,cursor,dest.y0,ratio,n,l,f'{part} {j+1}'))
        cursor+=widths[j]+gap
    yy=dest.y0+(bottom-top)*ratio
    if label:
        yy+=3;page.insert_font(fontname='AxisArial',fontfile=FONT)
        fs=13*ratio;font=fitz.Font(fontfile=FONT);ww=font.text_length(label,fontsize=fs)
        page.insert_text((dest.x0+(dest.width-ww)/2,yy+13*ratio),label,fontname='AxisArial',fontsize=fs,color=(.13,.13,.13))
        yy+=18*ratio
    if leg:
        yy+=3;place_fixed(page,doc,leg,dest.x0+(dest.width-leg.width*ratio)/2,yy,ratio,n,l,'shared legend')
    spacing.append(dict(figure=n,panel=l,part=part,outer_left_pt=placed[0].x0,outer_right_pt=placed[-1].x1,target_left_pt=dest.x0+gap,target_right_pt=dest.x1-gap,slot_left_pt=dest.x0,slot_right_pt=dest.x1,outer_left_gap_pt=placed[0].x0-dest.x0,outer_right_gap_pt=dest.x1-placed[-1].x1,gap_pt=gap,blocks=len(srcs),scale=ratio))

for n,plan in plans.items():
    if n==5:
        shutil.copy2(S/'baseline_Figure5.pdf',O/'Figure5.pdf')
        preserved=fitz.open(O/'Figure5.pdf')
        preserved[0].get_pixmap(matrix=fitz.Matrix(180/72,180/72),alpha=False).save(O/'Figure5.png')
        continue
    source_n={5:4,6:5}.get(n,n)
    ps=sorted([(k,v)for k,v in layout['panels'].items()if k.startswith(str(source_n))],key=lambda kv:(kv[1]['y'],kv[1]['x']))
    boxes={string.ascii_lowercase[i]:fitz.Rect(v['x']*54,v['y']*54,(v['x']+v['w'])*54,(v['y']+v['h'])*54)for i,(k,v)in enumerate(ps)}
    if n==1:boxes['legend']=fitz.Rect(0,972,1080,1053)
    doc=fitz.open(S/f'Figure{n}.pdf');page=doc[0]
    # Only panel marks and two descriptive titles are removed. Facet identifiers,
    # scientific variables, molecule identifiers and all measured marks remain.
    spans=[s for b in page.get_text('dict')['blocks']if 'lines'in b for l in b['lines']for s in l['spans']]
    removed=[]
    for s in spans:
        isletter=s['text'] in [x.upper()for x in boxes if len(x)==1] and s['size']>=15
        if isletter or s['text'] in ['Orthology features','Entity correspondence']:
            page.add_redact_annot(fitz.Rect(s['bbox']),fill=(1,1,1));removed.append(s['text'])
    page.apply_redactions(images=0,graphics=0)
    # Workflow letters are embedded in a Blender composite in three masters.
    page.draw_rect(fitz.Rect(0,0,23,28),fill=(1,1,1),color=None,overlay=True)
    # The secondary Blender workflow also contains its own old panel mark.
    if n==1:page.draw_rect(fitz.Rect(540,1053,563,1079),fill=(1,1,1),color=None,overlay=True)
    h=max(y+hh for l,x,y,w,hh in plan)
    out=fitz.open();p=out.new_page(width=1080,height=h)
    for l,x,y,w,hh in plan:
        box=boxes[l]; dest=fitz.Rect(x+12,y+31,x+w-12,y+hh-5)
        if l=='a':
            dest=fitz.Rect(24,y+3,1074,y+hh-3)
            put(p,doc,box,dest,n,l,'Blender workflow')
        elif l=='legend':put(p,doc,box,fitz.Rect(x+12,y+2,x+w-12,y+hh-2),n,l,'shared legend')
        elif n==1 and l=='b':
            # Separate the cohort grid from its quantitative/categorical legends.
            put(p,doc,(0,238,540,630),(x+12,y+31,x+w-12,y+hh-71),n,l,'cohort grid',align='top')
            put(p,doc,(0,630,540,702),(x+12,y+hh-68,x+w-12,y+hh-4),n,l,'legends')
        elif n==5 and l=='b':
            # Move the evidence-status legend upward after tightening the wheel.
            put(p,doc,(0,162,648,675),(x+12,y+31,x+w-12,y+hh-38),n,l,'wheel')
            put(p,doc,(0,675,648,756),(x+50,y+hh-32,x+w-50,y+hh-4),n,l,'status legend')
        elif n==6 and l=='b':
            # Four heatmap blocks retain every compound, with a separate key.
            for j in range(4):
                put(p,doc,(j*270,135,(j+1)*270,510),(x+j*270+8,y+31,x+(j+1)*270-8,y+hh-30),n,l,f'heatmap block {j+1}',align='top')
            put(p,doc,(0,510,1080,540),(x+270,y+hh-26,x+810,y+hh-4),n,l,'shared key')
        elif n==1 and l=='c':
            balanced_columns(p,doc,[(540,216,830,702),(830,216,1080,702)],dest,n,l,old_scale[n,l,'chart'])
        elif n==1 and l in ['d','e']:
            bx=0 if l=='d' else 540
            balanced_columns(p,doc,[(bx,702,bx+270,972),(bx+270,702,bx+540,972)],dest,n,l,old_scale[n,l,'chart'])
        elif n==2 and l in ['g','h']:
            edges=[0,236.35,390.35,540]if l=='g'else[540,785.15,929.3,1080]
            clips=[(edges[j],864,edges[j+1],1134)for j in range(3)]
            balanced_columns(p,doc,clips,dest,n,l,old_scale[n,l,'chart'],label='Joint − comparator'if l=='g'else'Joint − reference')
        elif n==2 and l=='i':
            edges=[0,174.35,295.3,416.3,540]
            clips=[(edges[j],1134,edges[j+1],1434)for j in range(4)]
            balanced_columns(p,doc,clips,dest,n,l,old_scale[n,l,'chart'],legend=(0,1435,540,1458))
        elif n==2 and l=='j':
            edges=[540,750.45,913.85,1080]
            clips=[(edges[j],1134,edges[j+1],1458)for j in range(3)]
            balanced_columns(p,doc,clips,dest,n,l,old_scale[n,l,'chart'],label='Training RNA mean')
        elif n==6 and l in ['f','g']:
            cols=2 if l=='f'else 3;bw=540/cols;offset=0 if l=='f'else 2
            ratio=min(value for (fig,letter,part),value in old_scale.items()if fig==6 and letter in ['f','g']and part.startswith('molecule '))
            for row in range(2):
                for col in range(cols):
                    src=tight(doc,fitz.Rect(box.x0+col*bw,756+row*135,box.x0+(col+1)*bw,756+(row+1)*135))
                    center=(offset+col+.5)*216
                    yy=y+31+row*(hh-36)/2
                    actual=place_fixed(p,doc,src,center-src.width*ratio/2,yy,ratio,n,l,f'molecule {row*cols+col+1}')
                    molecule_grid.append(dict(panel=l,row=row+1,column=offset+col+1,center_pt=center,scale=ratio,destination=list(actual)))
        elif n==6 and l=='h':
            edges=[0,252.4,459.1,665.9,872.7,1080]
            clips=[(edges[j],1026,edges[j+1],1273)for j in range(5)]
            balanced_columns(p,doc,clips,dest,n,l,old_scale[n,l,'chart'],label='UniProt residue',legend=(0,1274,1080,1296))
        else:put(p,doc,box,dest,n,l,align='top')
        if l!='legend':
            p.insert_font(fontname='PanelArial',fontfile=FONT)
            p.insert_text((x+12,y+24),l.upper(),fontname='PanelArial',fontsize=10/SCALE,color=(.15,.21,.24))
            records.append(dict(figure=n,panel=l,part='letter',x_mm=(x+12)*SCALE/72*25.4,y_mm=(y+24)*SCALE/72*25.4))
    # Use one physical publication width; the operation is uniformly scaled.
    final=fitz.open();f=final.new_page(width=1080*SCALE,height=h*SCALE);f.show_pdf_page(f.rect,out,0)
    final.save(O/f'Figure{n}.pdf',garbage=4,deflate=True)
    pix=f.get_pixmap(matrix=fitz.Matrix(180/72,180/72),alpha=False);pix.set_dpi(180,180);pix.save(O/f'Figure{n}.png')
    print(n,'height mm',h/6,'removed',removed,flush=True)
(O/'vector_layout.json').write_text(json.dumps(records,indent=2)+'\n')
(O/'panel_boxes.json').write_text(json.dumps({n:[dict(panel=l,x_mm=x/6,y_mm=y/6,width_mm=w/6,height_mm=h/6) for l,x,y,w,h in plan]for n,plan in plans.items()},indent=2)+'\n')

(O/'justification_geometry.json').write_text(json.dumps(spacing,indent=2)+'\n')

(O/'molecule_grid.json').write_text(json.dumps(molecule_grid,indent=2)+'\n')
