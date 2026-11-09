"""Build editable scientific workflow scenes for Blender and vector export."""
from pathlib import Path
import sys,json,os,math
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from workflow_primitives import Scene,BLUE,PURPLE,GREEN,INK
ASSETS=Path(os.environ.get('AD_WORKFLOW_ASSETS','/public/home/mengxl/dzy/ad_omics_v3/analysis/figure_build_20261031/data/workflow_assets'))
OUT=ROOT/'workflows';OUT.mkdir(exist_ok=True)
def new(n):
    s=Scene(n,1008,138,['Existing main-figure Blender workflow style'])
    s.e[0].update(x=12,y=5,w=22,h=25,size=21)
    return s
def tx(s,x,y,w,t,bold=False,size=14,h=20):s.text(x,y,w,h,t,size,bold)
def arrow(s,*pts,color=INK):s.arrow(*pts,stroke=color,lw=1.1,head=6)
def matrix(s,x,y,c=BLUE):
    start=len(s.e);s.matrix(x,y,4,3,6,[c,'#F1F5F6','#A8C4CF'])
    for e in s.e[start:]:
        if 'lw' in e:e['lw']=.8
def image(s,key,cx,cy,h):
    with Image.open(ASSETS/f'{key}.png') as im:
        box=im.getchannel('A').point(lambda x:255 if x>8 else 0).getbbox()
        scale=h/(box[3]-box[1]);s.asset(key,cx-(box[0]+box[2])/2*scale,cy-(box[1]+box[3])/2*scale,im.width*scale,im.height*scale)
def network(s,x,y,w=76,h=30):
    for a in range(2):
        for i in range(3):
            for j in range(3):s.line((x+a*w/2,y+i*h/2),(x+(a+1)*w/2,y+j*h/2),stroke='#C2CCD3',lw=.7)
    for a in range(3):
        for i in range(3):s.ellipse(x+a*w/2-4,y+i*h/2-4,8,8,[BLUE,PURPLE,GREEN][a],INK,.5)
def path(s,x,y):
    pts=[(x+t*2.4,y+9*math.sin(t/25*math.pi)) for t in range(26)]
    s.path(pts,stroke='#71899A',lw=1.3)
    for i in [0,8,17,25]:
        xx,yy=pts[i];s.ellipse(xx-4,yy-4,8,8,BLUE if i<10 else PURPLE,INK,.5)
    arrow(s,pts[-3],pts[-1])
def genes(s,x,y,c):
    s.rect(x,y,34,37,'#FFFFFF','#90A3AC',.7)
    for i,w in enumerate([21,16,24,12]):s.rect(x+5,y+7+i*7,w,3,c,'none',0)

s3=new(3)
s3.group='measured_perturbations';image(s3,'pbmc',58,57,32)
matrix(s3,102,35);matrix(s3,102,78,PURPLE)
tx(s3,32,106,152,'CD4 / myeloid Perturb-seq',size=12)
tx(s3,51,11,140,'Measured log2FC',True)
arrow(s3,(136,45),(167,45),(167,66),(193,66));arrow(s3,(136,88),(167,88),(167,66))
s3.group='source_state_shift';matrix(s3,230,35);matrix(s3,230,78,PURPLE)
tx(s3,194,11,150,'Shift source RNA',True)
tx(s3,270,27,82,'Baseline',size=12,h=14);tx(s3,270,90,82,'Shifted',size=12,h=14)
tx(s3,189,54,25,'Δ',True,size=18)
arrow(s3,(208,66),(214,66),(214,88),(220,88));arrow(s3,(242,56),(242,72))
arrow(s3,(255,45),(352,45));arrow(s3,(255,88),(352,88))
s3.group='frozen_model';path(s3,371,43);path(s3,371,86)
s3.lock(454,49,.48);tx(s3,353,10,181,'Frozen world model',True)
arrow(s3,(437,45),(541,45));arrow(s3,(437,88),(541,88))
s3.group='predicted_state_difference';matrix(s3,557,35,PURPLE);matrix(s3,557,78,PURPLE)
tx(s3,517,10,160,'Predicted brain states',True)
arrow(s3,(587,45),(631,45),(631,66),(676,66));arrow(s3,(587,88),(631,88),(631,66))
tx(s3,607,52,30,'−',True,size=22);matrix(s3,695,57,'#D49AA9')
tx(s3,661,88,112,'Brain response',True)
s3.group='gene_program_readouts';arrow(s3,(733,66),(810,66));genes(s3,839,47,GREEN);genes(s3,895,47,PURPLE)
tx(s3,799,10,177,'Genes and programs',True)
tx(s3,785,90,205,'GRN / BIN1 / ACP5 examples',size=12)
tx(s3,164,119,656,'Rest / stimulation in CD4 T cells; untreated / LPS in myeloid cells',size=12,h=16)

s4=new(4)
s4.group='brain_isolated_source';image(s4,'brain',59,58,34);matrix(s4,140,49,BLUE)
tx(s4,33,12,179,'Brain CD45+ populations',True)
tx(s4,106,77,107,'Phagocyte RNA',size=13)
arrow(s4,(84,58),(124,58))
s4.group='frozen_predictions';arrow(s4,(173,58),(250,58));network(s4,266,43,76,30);s4.lock(358,45,.48)
tx(s4,232,12,179,'Frozen world model',True)
arrow(s4,(386,58),(433,58));matrix(s4,451,49,PURPLE)
tx(s4,414,77,122,'Model prediction',size=13)
s4.group='fold_fitted_readout';arrow(s4,(483,58),(556,58))
for i,c in enumerate([BLUE,PURPLE,GREEN]):
    for j in range(3):s4.ellipse(577+i*18,45+j*11,7,7,c,INK,.5)
tx(s4,549,12,122,'Gene-wise ridge',True)
tx(s4,552,79,136,'Fit in training fold',size=13)
arrow(s4,(154,71),(154,105),(535,105),(535,68),(574,68),color='#3C947C')
tx(s4,290,90,158,'Source state also enters',size=12,h=14)
s4.group='response_reference';arrow(s4,(631,58),(692,58),(692,45),(711,45));arrow(s4,(692,58),(692,84),(711,84));matrix(s4,721,36,GREEN);matrix(s4,721,75,PURPLE)
tx(s4,756,30,55,'LPS',size=13);tx(s4,750,70,66,'Saline',size=13)
tx(s4,690,12,145,'Predicted microglia',True)
arrow(s4,(817,45),(841,45),(841,65),(865,65));arrow(s4,(817,84),(841,84),(841,65))
tx(s4,820,50,28,'−',True,size=22)
s4.group='response_programs';genes(s4,884,47,GREEN);genes(s4,935,47,PURPLE)
tx(s4,853,12,148,'Microglial response',True)
tx(s4,848,90,158,'Genes and programs',size=13)
tx(s4,38,120,932,'Systemic LPS / saline · reciprocal DMSO / inhibitor split · 4 train / 4 test animals per fold',size=12,h=16)

scenes=[dict(figure=s.n,width=s.w,height=s.h,reference_panels=s.refs,elements=s.e) for s in [s3,s4]]
issues=[]
for sc in scenes:
    c=canvas.Canvas(str(OUT/f'{sc["figure"]}a_vector.pdf'),pagesize=(sc['width'],sc['height']))
    for e in sc['elements']:
        typ=e['type'];fill=e.get('fill','none');stroke=e.get('stroke','none')
        c.setLineWidth(e.get('lw',.7));c.setFillColor(HexColor(fill) if fill!='none' else HexColor('#FFFFFF'));c.setStrokeColor(HexColor(stroke) if stroke!='none' else HexColor('#FFFFFF'))
        if typ in ['rect','ellipse']:
            if typ=='rect':c.rect(e['x'],sc['height']-e['y']-e['h'],e['w'],e['h'],fill=fill!='none',stroke=stroke!='none')
            else:c.ellipse(e['x'],sc['height']-e['y']-e['h'],e['x']+e['w'],sc['height']-e['y'],fill=fill!='none',stroke=stroke!='none')
        elif typ=='path':
            p=c.beginPath();p.moveTo(e['pts'][0][0],sc['height']-e['pts'][0][1])
            for x,y in e['pts'][1:]:p.lineTo(x,sc['height']-y)
            if e['closed']:p.close()
            c.drawPath(p,fill=fill!='none',stroke=stroke!='none')
        elif typ=='image':c.drawImage(str(ASSETS/f'{e["asset"]}.png'),e['x'],sc['height']-e['y']-e['h'],e['w'],e['h'],mask='auto')
        elif typ=='text':
            font='Helvetica-Bold' if e['bold'] else 'Helvetica';c.setFont(font,e['size']);c.setFillColor(HexColor(e['color']))
            if stringWidth(e['text'],font,e['size'])>e['w']+2:issues.append([sc['figure'],e['text'],e['w']])
            x=e['x'] if e['align']=='left' else e['x']+e['w']/2
            (c.drawString if e['align']=='left' else c.drawCentredString)(x,sc['height']-e['y']-e['h']/2-e['size']*.34,e['text'])
    c.save()
(ROOT/'workflow_scenes.json').write_text(json.dumps(scenes,indent=2)+'\n')
(OUT/'text_fit.json').write_text(json.dumps(issues,indent=2)+'\n')
print('Workflow text fit flags:',issues)
