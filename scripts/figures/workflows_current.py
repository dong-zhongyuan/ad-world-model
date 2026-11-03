# Reuse the copied drawing primitives and per-element renderer; current scenes redesigned.
from pathlib import Path
import json,copy,math,os
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from workflow_primitives import Scene,BLUE as B,PURPLE as P,GREEN as G,GOLD as Y,INK
from PIL import Image
from molecular_glyphs import molecule
import pandas as pd
from paths import REPO_ROOT as ROOT, FIGURE_WORK as W
PANEL_BUILD=W/'build'
ASSETS=Path(os.environ.get('AD_WORKFLOW_ASSETS',str(W/'workflow_assets')))
out=[]
def setup(n):
 s=Scene(n,720,360,['Reference_1C','Reference_1D','Reference_2A','Reference_3A','Reference_4A','Reference_6A','Reference_7A','Reference_S1D']);s.e[0].update(size=10,x=2,y=2,w=12,h=12);return s
def tx(s,x,y,w,h,t,b=False,align='center'):s.text(x,y,w,h,t,13 if b else 12,b,align)
def arrow(s,*p):s.arrow(*p,lw=.85,head=5)
def supervision(s,*points):
 # Explicit segments retain dashed supervision in both PDF and editable PPT.
 for (x0,y0),(x1,y1) in zip(points,points[1:]):
  length=math.hypot(x1-x0,y1-y0)
  for start in range(0,int(length),7):
   a=start/length;b=min(start+4,length)/length
   s.line((x0+a*(x1-x0),y0+a*(y1-y0)),(x0+b*(x1-x0),y0+b*(y1-y0)),stroke='#71818C',lw=.85)
 x,y=points[-1];px,py=points[-2];angle=math.atan2(y-py,x-px)
 s.path([(x,y),(x-5*math.cos(angle)+2*math.sin(angle),y-5*math.sin(angle)-2*math.cos(angle)),(x-5*math.cos(angle)-2*math.sin(angle),y-5*math.sin(angle)+2*math.cos(angle))],fill='#71818C',stroke='#71818C',lw=.4,closed=True)
def box(s,x,y,w,h,t,c=B):s.rect(x,y,w,h,c,INK,.65);tx(s,x+5,y+3,w-10,h-6,t,True)
def region(s,x,y,w,h,c='#F5F8FA'):s.rect(x,y,w,h,c,'#D8DFE3',.55)
def mat(s,x,y,c=B,cols=5,rows=4,cell=8):
 start=len(s.e);s.matrix(x,y,cols,rows,cell,[c,'#F5F7F8','#A9BFC8'])
 for e in s.e[start:]:
  if 'lw' in e:e['lw']=.65

def asset(s,key,cx,cy,h=65):
 with Image.open(ASSETS/f'{key}.png') as im:
  bb=im.convert('RGBA').getchannel('A').point(lambda a:255 if a>8 else 0).getbbox();sc=h/(bb[3]-bb[1]);s.asset(key,cx-(bb[0]+bb[2])/2*sc,cy-(bb[1]+bb[3])/2*sc,im.width*sc,im.height*sc)
def lock(s,x,y):s.lock(x,y,.35)
def seeds(s,x,y,vals,fill=G,cross=None):
 for i,v in enumerate(vals):
  xx=x+i*28;s.rect(xx,y,24,20,fill,'#95A8B2',.5);tx(s,xx,y,24,20,str(v))
  if v==cross:s.line((xx,y),(xx+24,y+20),stroke='#B85865',lw=1.1)
def finish(s):
 # Reflow native objects into a generous portrait-ratio composition. Typography
 # is reset independently, rather than scaling a rendered panel image.
 spec=json.loads((W/'page_layout.json').read_text());key='1h' if s.n==6 else str(s.n)+'a';dim=spec['panels'][key]
 nw=dim['w']*spec['unit_pt'];nh=dim['h']*spec['unit_pt'];sx=nw/s.w;sy=nh/s.h
 for e in s.e:
  for k in ['x','w']:
   if k in e:e[k]*=sx
  for k in ['y','h']:
   if k in e:e[k]*=sy
  if 'pts' in e:e['pts']=[(x*sx,y*sy) for x,y in e['pts']]
 s.w=nw;s.h=nh
 for e in s.e:
  if e['type']=='text':e['size']=16 if e['group']=='frame' else (13 if e['bold']else 12)
 out.append(dict(figure=s.n,width=s.w,height=s.h,reference_panels=s.refs,elements=s.e))
# Compact scientific process glyphs. They carry no empirical plotted values.
def network(s,x,y,w=100,h=54):
 for a in range(2):
  for i in range(3):
   for j in range(3):s.line((x+a*w/2,y+i*h/2),(x+(a+1)*w/2,y+j*h/2),stroke='#C2CCD3',lw=.6)
 for a in range(3):
  for i in range(3):s.ellipse(x+a*w/2-4,y+i*h/2-4,8,8,[B,P,G][a],INK,.5)
def vector(s,x,y,c=B,n=8):
 for i in range(n):s.rect(x+i*9,y,7,10,c if i%3 else'#DCE6E9','none',0)
def people(s,x,y,n=3,c=B,step=17):
 for i in range(n):
  xx=x+i*step;s.ellipse(xx+4,y,7,7,c,'#78949E',.5);s.rect(xx+2,y+9,11,16,c,'#78949E',.5)
def genes(s,x,y,w=49,h=62,c=G,rows=5,offset=0):
 s.rect(x,y,w,h,'#FFFFFF','#90A3AC',.55)
 for i in range(rows):
  s.rect(x+5,y+7+i*(h-12)/rows,w-10,3,c if(i+offset)%3 else'#DEE6E8','none',0)
def dna(s,x,y,w=55,h=52):
 a=[];b=[]
 for i in range(21):
  t=i/20;xx=x+t*w;yy=y+h/2+math.sin(t*math.tau)*h*.36;a.append((xx,yy));b.append((xx,2*y+h-yy))
 s.path(a,stroke='#609BB9',lw=1.4);s.path(b,stroke='#A995BF',lw=1.4)
 for i in range(2,20,3):s.line(a[i],b[i],stroke='#C9A66C',lw=.85)
def matchcloud(s,x,y):
 # Two population distributions; there are no cell-to-cell correspondences.
 aa=[(0,0),(12,15),(-7,21),(16,-8),(-13,-9)];bb=[(59,2),(71,17),(54,23),(77,-5),(50,-10)]
 for i,(a,b) in enumerate(zip(aa,bb)):
  for pt,c in [(a,B),(b,P)]:s.ellipse(x+pt[0]-3,y+pt[1]-3,6,6,c,'#66858F',.4)
def nullsets(s,x,y):
 genes(s,x,y,37,49,G,5)
 arrow(s,(x+44,y+24),(x+62,y+24))
 for j in range(3):genes(s,x+69+j*28,y+(j%2)*9,25,40,B,5,j)
def metric_glyph(s,key,x,y):
 if key=='MMD':matchcloud(s,x+17,y+16)
 elif key=='MSE':
  vector(s,x+2,y+1,B,5);vector(s,x+2,y+22,G,5);s.line((x+57,y+4),(x+64,y+4),(x+64,y+29),(x+57,y+29),stroke=INK,lw=.7)
 elif key=='Direction':
  for i,v in enumerate([1,-1,1,-1]):
   xx=x+6+i*18;s.line((xx,y+15),(xx,y+15-v*12),stroke=['#B66B79','#5B99B0'][v<0],lw=3)
  s.line((x,y+15),(x+72,y+15),stroke='#A0ACB3',lw=.5)
 elif key=='Blood':
  vector(s,x+4,y+1,B,5);lock(s,x+58,y-3);arrow(s,(x+23,y+16),(x+23,y+36));vector(s,x+4,y+38,P,5)
def barcode(s,x,y,w=70,h=18):
 for i in range(25):s.rect(x+i*w/25,y,(1+(i%3==0))*w/70,h,INK,'none',0)
def spatialgrid(s,x,y):
 for r in range(4):
  for c in range(6):s.ellipse(x+c*10+(r%2)*4,y+r*9,4,4,[B,G,P,'#DBBEC6'][(r+c)%4],'none',0)

# AD process diagrams: each branch, connector and glyph represents an actual operation.
def ad_scene(n,w=1080,h=360):
 s=setup(n);s.w=w;s.h=h;return s

def statepath(s,x,y,w=150,h=45):
 pts=[(x+w*t/30,y+h*math.sin(t/30*math.pi)*.7)for t in range(31)]
 s.path(pts,stroke='#71899A',lw=1.4)
 for t in [0,10,20,30]:
  xx,yy=pts[t];s.ellipse(xx-5,yy-5,10,10,[B,'#8DA7BC','#9D9FBC',P][t//10],INK,.5)
 arrow(s,pts[-3],pts[-1])

def routes_glyph(s,x,y):
 labels=['B','CD4T','CD8T','Mono','NK'];ys=[y+q*7 for q in range(5)]
 for lab,yy in zip(labels,ys):
  s.ellipse(x,yy,3,3,B,'#4F8DA7',.3);tx(s,x-26,yy-3,23,9,lab,False,'right');s.line((x+3,yy+1.5),(x+25,y+9),stroke='#A4B9C6',lw=.45)
 s.line((x+3,ys[3]+1.5),(x+25,y+26),stroke='#A4B9C6',lw=.45)
 for yy,lab in [(y+7,'Micro'),(y+24,'Oligo')]:s.ellipse(x+25,yy,4,4,P,'#8E7EB4',.3);tx(s,x+31,yy-3,25,9,lab,False,'left')

# The same scientific glyphs are reflowed at final paper dimensions.
s=ad_scene(1,504,100.8);s.group='species_inputs'
for yy,lab,c in [(23,'Human',B),(61,'Mouse',Y)]:
 asset(s,'pbmc',22,yy,19);tx(s,3,yy+10,38,10,lab,True);mat(s,48,yy-8,c,4,4,4);arrow(s,(66,yy),(77,yy),(77,40),(85,40))
s.group='orthology';genes(s,87,27,22,24,G,4);tx(s,74,54,52,10,'Orthogroups',True);arrow(s,(111,40),(135,40))
s.group='world_model';network(s,140,31,40,18);tx(s,134,53,52,10,'Encoder',True);arrow(s,(185,40),(202,40));statepath(s,207,38,48,15);tx(s,202,55,59,10,'Neural ODE',True);arrow(s,(260,40),(277,40));mat(s,282,30,P,5,5,4);tx(s,269,54,49,10,'Brain RNA',True)
s.group='brain_supervision';asset(s,'brain',292,13,15);supervision(s,(292,23),(292,29))
s.group='protein_supervision';arrow(s,(160,50),(160,75),(191,75));vector(s,196,72,G,4);tx(s,187,85,70,10,'Protein head',True);asset(s,'cite_cell',128,82,18);supervision(s,(139,82),(184,82));tx(s,89,89,92,9,'CITE-seq RNA + ADT')
s.group='network';arrow(s,(306,40),(339,40));routes_glyph(s,373,20);tx(s,334,57,113,10,'Six blood-brain routes',True)
s.group='independent_genetics';dna(s,282,79,27,14);asset(s,'protein',338,81,18);asset(s,'brain',394,81,18);arrow(s,(311,81),(326,81));arrow(s,(350,81),(382,81));tx(s,272,91,46,9,'cis-pQTL');tx(s,316,91,45,9,'Protein');tx(s,373,91,44,9,'AD risk')
s.group='evidence_join';arrow(s,(430,40),(455,40),(455,73));arrow(s,(407,81),(442,81));asset(s,'protein',455,81,17);arrow(s,(467,81),(482,81));molecule(s,492,79,5);tx(s,433,91,42,9,'Target');tx(s,477,91,26,9,'Drug')
finish(s)

s=ad_scene(2,504,75.6);s.group='training_inputs'
for y,lab in [(20,'Human'),(48,'Mouse')]:asset(s,'pbmc',20,y,17);tx(s,2,y+9,36,9,lab,True)
network(s,80,14,36,14);network(s,80,42,36,14);tx(s,121,13,61,10,'Human model',True);tx(s,121,42,61,10,'Joint model',True)
arrow(s,(33,20),(74,20));arrow(s,(33,25),(60,25),(60,48),(74,48));arrow(s,(33,48),(74,48));tx(s,38,64,154,9,'1,500 updates; exposure control 3,000')
s.group='frozen_predictions';arrow(s,(183,20),(199,20),(199,39));arrow(s,(183,47),(199,47),(199,39));lock(s,195,28);arrow(s,(204,39),(220,39));s.line((221,19),(221,58),stroke=INK,lw=.5)
s.group='brain_test';asset(s,'brain',241,19,19);tx(s,224,30,43,9,'Brain test',True)
for i,lab in enumerate(['AD10','AD4','NC18','NC7']):people(s,280+i*28,11,1,P);tx(s,271+i*28,38,34,9,lab)
arrow(s,(390,20),(410,20));matchcloud(s,415,18);tx(s,414,40,86,9,'Distributions',True)
s.group='protein_test';asset(s,'cite_cell',241,58,19)
for i,lab in enumerate(['P6','P7','P8']):mat(s,279+i*35,53,G,3,3,4);tx(s,271+i*35,66,31,9,lab)
arrow(s,(389,59),(410,59));mat(s,425,52,G,3,3,4);mat(s,450,52,Y,3,3,4);tx(s,403,66,99,9,'Predicted / observed')
finish(s)

s=ad_scene(3,504,75.6);s.group='input_clamp'
asset(s,'pbmc',20,18,17);mat(s,47,12,B,5,4,4);mat(s,47,47,B,5,4,4);s.rect(47,53,20,4,'#FFFFFF','#B64E65',.6);arrow(s,(57,30),(57,43));tx(s,5,64,84,9,'Baseline / clamped RNA')
s.group='paired_model_queries';arrow(s,(71,20),(91,20));arrow(s,(71,55),(91,55));statepath(s,97,19,51,13);statepath(s,97,54,51,13);tx(s,91,36,66,10,'Frozen ODE',True)
for yy in [12,47]:mat(s,171,yy,P,5,4,4)
arrow(s,(153,20),(167,20));arrow(s,(153,55),(167,55));tx(s,160,65,47,9,'Brain RNA')
s.group='difference';arrow(s,(194,20),(212,20),(212,38),(230,38));arrow(s,(194,55),(212,55),(212,38));tx(s,205,30,13,14,'−',True);mat(s,234,29,'#B8758A',5,4,4);tx(s,218,49,58,10,'Response',True)
s.group='matched_null';genes(s,283,48,13,17,B,3);genes(s,300,48,13,17,B,3);genes(s,317,48,13,17,B,3);tx(s,273,65,67,9,'Matched null');supervision(s,(282,39),(261,39));tx(s,274,15,67,10,'Reference sets')
s.group='routes';arrow(s,(260,32),(339,32));routes_glyph(s,369,16);arrow(s,(425,33),(443,33));genes(s,450,21,19,25,G,4);genes(s,477,21,19,25,P,4);tx(s,433,51,70,18,'Genes + programs',True)
finish(s)

s=ad_scene(4,504,75.6);s.group='network_support';routes_glyph(s,36,13);arrow(s,(97,30),(117,30));tx(s,106,39,50,10,'Top 50%',True)
s.group='protein_mr';dna(s,13,58,28,12);asset(s,'protein',68,58,16);asset(s,'brain',101,58,16);arrow(s,(42,58),(55,58));arrow(s,(80,58),(90,58));tx(s,4,66,112,9,'cis-pQTL → protein → AD')
s.group='outcome_evidence';arrow(s,(115,58),(151,58));mat(s,162,45,G,4,4,4);mat(s,208,45,P,4,4,4);tx(s,147,64,49,9,'Bellenguez');tx(s,197,64,40,9,'FinnGen');tx(s,148,12,91,10,'Protein-to-AD MR',True)
s.group='clinical_followup';tx(s,242,2,60,24,'ADNI\nsurvival',True)
s.group='identity';genes(s,260,24,20,25,G,4);tx(s,243,54,54,18,'Gene / protein\nidentity',True);arrow(s,(137,29),(257,29));arrow(s,(225,52),(257,40))
s.group='target_groups';arrow(s,(283,35),(306,35),(306,20),(321,20));arrow(s,(306,35),(306,57),(321,57))
for yy,lab,targets,col in [(17,'Both outcomes',['BIN1','CR1','PILRB','GRN','BLNK'],G),(52,'Union only',['CD55','ACP5','ST13','NESP55','BST2'],P)]:
 tx(s,323,yy-15,178,10,lab,True)
 for j,t in enumerate(targets):
  xx=331+j*36;s.ellipse(xx,yy,8,8,col,'#77989C',.4);s.ellipse(xx+3,yy+2,2.5,2.5,'#FFFFFF','none',0);tx(s,xx-12,yy+10,33,10,t,True)
finish(s)

s=ad_scene(5,504,63);s.group='compound_library';molecule(s,23,17,7);tx(s,2,30,45,9,'2,297 drugs');network(s,66,12,32,12);arrow(s,(37,18),(60,18));tx(s,50,1,66,9,'Dual encoders',True)
s.group='pocket_library';asset(s,'protein',23,49,18);network(s,66,43,32,12);arrow(s,(37,49),(60,49));tx(s,1,53,46,9,'323 pockets')
s.group='similarities';arrow(s,(103,18),(118,18),(118,34),(134,34));arrow(s,(103,49),(118,49),(118,34));
for i in [2,1,0]:mat(s,141+i*3,22-i*3,G,6,6,4)
tx(s,128,51,72,9,'Six-model scores',True)
s.group='parent_aggregation';arrow(s,(174,34),(208,34))
for j,ww in enumerate([30,25,20,15,10]):s.rect(217,18+j*6,ww,3,G,'none',0)
tx(s,199,51,72,9,'Parent ranking',True)
s.group='fold_agreement';arrow(s,(249,25),(285,25));
for j in range(6):s.rect(296+j*7,17,4,4,G,'#809E9A',.3)
tx(s,281,4,74,9,'Agreement',True)
s.group='chemical_series';arrow(s,(249,39),(285,39));molecule(s,306,43,5);molecule(s,330,43,5);tx(s,282,51,74,9,'Chemical series',True)
s.group='shortlisted_poses';arrow(s,(350,25),(377,25),(377,34),(417,34));arrow(s,(350,43),(377,43),(377,34));asset(s,'protein',450,28,33);molecule(s,458,34,5);tx(s,408,51,95,9,'Exact-state poses',True)
finish(s)

s=ad_scene(6,252,126);s.e[0]['text']='H';s.group='encoder';asset(s,'pbmc',25,26,22);tx(s,2,42,47,10,'Blood RNA',True);arrow(s,(43,26),(64,26));network(s,70,17,40,18);tx(s,60,43,60,10,'RNA encoder',True)
s.group='dynamics';arrow(s,(115,26),(139,26));statepath(s,144,25,43,14);tx(s,130,44,66,10,'Neural ODE',True);arrow(s,(192,26),(213,26));asset(s,'brain',232,26,20);tx(s,207,43,43,10,'Brain RNA',True)
s.group='protein_head';arrow(s,(124,26),(124,54),(86.5,54),(86.5,60));network(s,67,65,39,17);tx(s,57,91,61,8,'Protein head',True);arrow(s,(111,73),(157,73));vector(s,165,68,G,7);tx(s,153,91,97,8,'224 ADT channels',True)
s.group='supervision';asset(s,'cite_cell',24,75,21);supervision(s,(38,73.5),(61,73.5));tx(s,2,88,49,13,'CITE-seq\nRNA + ADT')
# Named donor icons identify the exact eight CITE-seq donors. The bracket
# denotes dataset membership; partition labels distinguish training and evaluation.
s.group='citeseq_donors';donor_train='#D7E5E1';donor_val='#F0E4C6';donor_test='#EBD0D7'
for label,cx in [('Train',75),('Validation',140),('Held-out',209)]:tx(s,cx-24,100,48,6,label)
for i,(cx,fill) in enumerate([(45,donor_train),(65,donor_train),(85,donor_train),(105,donor_train),(140,donor_val),(184,donor_test),(209,donor_test),(234,donor_test)],1):
 start=len(s.e);people(s,0,0,1,fill)
 for e in s.e[start:]:
  e['x']=cx+(e['x']-7.5)*.34;e['y']=107+e['y']*.34;e['w']*=.34;e['h']*=.34;e['lw']=.4
 tx(s,cx-8,117,16,6,'P'+str(i))
s.line((14,83),(6,83),(6,124),(244,124),(244,121),stroke='#8B9DA7',lw=.5)
finish(s)

# Original per-object renderer retained; text reflow records real overflow instead of shrinking.
def lines_for(e):
 font='Helvetica-Bold'if e.get('bold')else'Helvetica';size=e['size'];w=e['w'];lines=[]
 for paragraph in e['text'].split('\n'):
  cur=''
  for word in paragraph.split(' '):
   trial=(cur+' '+word).strip()
   if stringWidth(trial,font,size)>w and cur:lines.append(cur);cur=word
   else:cur=trial
  lines.append(cur)
 return lines
issues=[]
for sc in out:
 if os.environ.get('FIGURE') and sc['figure']!=int(os.environ['FIGURE']):continue
 c=canvas.Canvas(str(PANEL_BUILD/f'{str(sc["figure"])+"a" if sc["figure"]!=6 else "1h"}.pdf'),pagesize=(sc['width'],sc['height']));c.setTitle(f'{sc["figure"]}a')
 for e in sc['elements']:
  typ=e['type'];fill=e.get('fill','none');stroke=e.get('stroke','none')
  c.setLineWidth(e.get('lw',.7));c.setFillColor(HexColor(fill)if fill!='none'else HexColor('#FFFFFF'));c.setStrokeColor(HexColor(stroke)if stroke!='none'else HexColor('#FFFFFF'))
  if typ in ('rect','ellipse'):
   fn=c.rect if typ=='rect'else c.ellipse
   args=(e['x'],sc['height']-e['y']-e['h'],e['w'],e['h'])if typ=='rect'else(e['x'],sc['height']-e['y']-e['h'],e['x']+e['w'],sc['height']-e['y'])
   fn(*args,fill=int(fill!='none'),stroke=int(stroke!='none'))
  elif typ=='path':
   p=c.beginPath();p.moveTo(e['pts'][0][0],sc['height']-e['pts'][0][1]);[p.lineTo(x,sc['height']-y)for x,y in e['pts'][1:]]
   if e.get('closed'):p.close()
   c.drawPath(p,fill=int(fill!='none'),stroke=int(stroke!='none'))
  elif typ=='image':
   paths=list(ASSETS.glob('*'+e['asset']+'*.png'))
   if not paths:raise RuntimeError(e['asset'])
   c.drawImage(str(paths[0]),e['x'],sc['height']-e['y']-e['h'],e['w'],e['h'],preserveAspectRatio=True,anchor='c',mask='auto')
  elif typ=='text':
   font='Helvetica-Bold'if e.get('bold')else'Helvetica';size=e['size'];c.setFont(font,size);c.setFillColor(HexColor(e['color']));lines=lines_for(e);lh=size*1.17;total=lh*len(lines)
   if total>e['h']+4 and e['text'].strip():issues.append((sc['figure'],e['text'],round(total-e['h'],1)))
   base=sc['height']-e['y']-(e['h']-total)/2-size*.85
   for j,t in enumerate(lines):
    x=e['x']if e.get('align')=='left'else e['x']+e['w']/2
    (c.drawString if e.get('align')=='left'else c.drawCentredString)(x,base-j*lh,t)
 c.showPage();c.save()
for s in out:
 for e in s['elements']:
  if e['type']=='text':e['text']='\n'.join(lines_for(e))
(W/'workflow_scenes.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
(W/'workflow_text_fit.json').write_text(json.dumps(issues,ensure_ascii=False,indent=2));print('Text fit flags',issues)
