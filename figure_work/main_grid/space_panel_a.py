"""Reserve a separate left gutter for A, preserving vector marks and aspect ratio."""
from pathlib import Path
import json,hashlib,sys
import pymupdf as fitz
import numpy as np
P=Path(__file__).resolve().parent
S=Path(sys.argv[1])if len(sys.argv)>1 else P/'output'
O=Path(sys.argv[2])if len(sys.argv)>2 else P/'output';O.mkdir(parents=True,exist_ok=True)
mm=72/25.4
heights={1:212/6,2:158/6,3:24,4:24,5:164/6,6:140/6}
font=next(p for p in [Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf'),Path('/public/home/mengxl/dzy/tools/fonts/Arial Bold.ttf')]if p.exists())
records=[]
for n in range(1,7):
    file=S/f'Figure{n}.pdf';source_digest=hashlib.sha256(file.read_bytes()).hexdigest();doc=fitz.open(file);pg=doc[0];h=heights[n]*mm
    if doc.metadata.get('keywords')=='Panel A clear left gutter':
        raise RuntimeError('Input has already been spaced; regenerate original layout first')
    spans=[s for b in pg.get_text('dict',flags=0)['blocks'] for line in b.get('lines',[]) for s in line['spans']]
    labels=[s for s in spans if s['text']=='A'and abs(s['size']-10)<.05 and s['origin'][1]<7*mm]
    assert len(labels)==1,(n,labels)
    old_label=labels[0]
    pg.add_redact_annot(fitz.Rect(old_label['bbox']),fill=None,cross_out=False)
    pg.apply_redactions(images=0,graphics=0)
    clip=fitz.Rect(0,0,pg.rect.width,h)
    pix=pg.get_pixmap(clip=clip,matrix=fitz.Matrix(3,3),alpha=False)
    a=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
    yy,xx=np.where(a.min(axis=2)<245)
    src=fitz.Rect(max(0,(pix.x+xx.min())/3-1),max(0,(pix.y+yy.min())/3-1),min(pg.rect.width,(pix.x+xx.max()+1)/3+1),min(h,(pix.y+yy.max()+1)/3+1))
    available=fitz.Rect(10*mm,1*mm,178*mm,h-1*mm)
    ratio=min(available.width/src.width,available.height/src.height)
    w,hh=src.width*ratio,src.height*ratio
    dst=fitz.Rect(available.x0,available.y0+(available.height-hh)/2,available.x0+w,available.y0+(available.height+hh)/2)
    out=fitz.open();p=out.new_page(width=pg.rect.width,height=pg.rect.height)
    rest=fitz.Rect(0,h,pg.rect.width,pg.rect.height)
    p.show_pdf_page(rest,doc,0,clip=rest)
    p.show_pdf_page(dst,doc,0,clip=src,keep_proportion=True)
    p.insert_font(fontname='PanelAClearance',fontfile=str(font))
    p.insert_text((2*mm,4*mm),'A',fontsize=10,fontname='PanelAClearance',color=(.15,.21,.24))
    meta=doc.metadata.copy();meta['keywords']='Panel A clear left gutter';out.set_metadata(meta)
    pending=O/f'Figure{n}.pending.pdf';out.save(pending,garbage=4,deflate=True);pending.replace(O/f'Figure{n}.pdf')
    p.get_pixmap(clip=clip,matrix=fitz.Matrix(3,3),alpha=False).save(O/f'Figure{n}A_preview.png')
    af=fitz.Font(fontfile=str(font))
    newlabel={'bbox':(2*mm,4*mm-10*af.ascender,2*mm+af.text_length('A',fontsize=10),4*mm-10*af.descender)}
    record=dict(figure=n,source_sha256=source_digest,panel_height_mm=heights[n],old_label_bbox_mm=[v/mm for v in old_label['bbox']],old_body_bbox_mm=[v/mm for v in src],new_label_bbox_mm=[v/mm for v in newlabel['bbox']],new_body_bbox_mm=[v/mm for v in dst],label_origin_mm=[2,4],horizontal_clearance_mm=(dst.x0-newlabel['bbox'][2])/mm,uniform_scale=ratio)
    assert record['horizontal_clearance_mm']>5
    records.append(record);print('Spaced A',n,'gap mm',round(record['horizontal_clearance_mm'],2),'scale',round(ratio,3),flush=True)
(O/'panel_a_geometry.json').write_text(json.dumps(records,indent=2)+'\n')
