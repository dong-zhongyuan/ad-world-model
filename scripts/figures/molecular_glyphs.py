"""Editable molecular schematics; shapes are illustrative, not molecular coordinates."""
import json,math
from pathlib import Path
W=Path(__file__).resolve().parents[1]
INK='#48626D';BLUE='#76A5BF';GOLD='#DCB77A';GREEN='#86B7A8';PURPLE='#A895BC'
def antibody(s,cx,cy,h=60):
    a=json.loads((W/'assets/bioicons/antibody_paths.json').read_text());x0,y0,x1,y1=a['box'];k=h/(y1-y0)
    for e in a['paths']:
        pts=[(cx+(x-(x0+x1)/2)*k,cy+(y-(y0+y1)/2)*k)for x,y in e['pts']]
        fill=e['fill'];stroke=e['stroke']
        if fill not in ['none','#FFFFFF']:
            rgb=[int(fill[i:i+2],16)for i in (1,3,5)];fill=GOLD if rgb[0]>rgb[2]+20 else BLUE
        s.path(pts,fill,INK if stroke!='none'else'none',max(.35,(e['lw'] or 0)*k),e['closed'] or fill!='none')
def membrane(s,x,y,w=100):
    # Heads and acyl tails of two leaflets remain independently editable.
    for j in range(int(w/7)):
        xx=x+j*7
        for yy,sgn in [(y,1),(y+17,-1)]:
            s.line((xx-1,yy+sgn*2),(xx-2,yy+sgn*7),stroke='#ADC8C0',lw=.65)
            s.line((xx+1,yy+sgn*2),(xx+2,yy+sgn*7),stroke='#ADC8C0',lw=.65)
            s.ellipse(xx-2.6,yy-2.6,5.2,5.2,'#D4E7DE','#97B9AB',.4)
def receptor(s,cx,y,kind='CD22',scale=1):
    # y is the top membrane leaflet. Domain architecture is schematic.
    k=scale
    s.rect(cx-3*k,y-7*k,6*k,31*k,BLUE,INK,.65)
    s.line((cx,y+24*k),(cx+5*k,y+34*k),(cx-3*k,y+40*k),stroke=INK,lw=1)
    if kind=='CD22':
        for j in range(7):
            yy=y-(j+1)*9*k-5*k;xx=cx+math.sin(j*.6)*3*k
            s.ellipse(xx-5*k,yy-5*k,10*k,8*k,'#A6C9D8'if j<6 else GOLD,INK,.55)
            s.line((xx-2*k,yy-2*k),(xx+2*k,yy+1*k),stroke='#FFFFFF',lw=.6)
    elif kind=='CD40':
        for j in range(4):
            yy=y-(j+1)*13*k
            s.ellipse(cx-7*k,yy-6*k,14*k,11*k,GREEN,INK,.6)
            s.ellipse(cx-4*k,yy-2*k,2.6*k,2.6*k,GOLD,INK,.2);s.ellipse(cx+2*k,yy+1*k,2.6*k,2.6*k,GOLD,INK,.2)
            s.line((cx-2*k,yy),(cx+2*k,yy+2*k),stroke='#CDA157',lw=.65)
    else:
        for sg in [-1,1]:
            s.line((cx+sg*4*k,y+9*k),(cx+sg*9*k,y-24*k),stroke=INK,lw=3*k)
            s.ellipse(cx+sg*12*k-12*k,y-54*k,24*k,32*k,PURPLE,INK,.7)
            s.ellipse(cx+sg*12*k-7*k,y-50*k,12*k,12*k,'#D7CBE2','none',0)
        s.line((cx-8*k,y-21*k),(cx+8*k,y-21*k),stroke=GOLD,lw=2*k)
def protein(s,cx,cy,w=65,h=50,c=BLUE,cleft=False):
    pts=[]
    for j in range(100):
        a=j*math.tau/100;r=1+.08*math.sin(5*a)+.05*math.cos(9*a)
        if cleft:r*=1-.42*math.exp(-((a-math.pi/2)/.42)**2)
        pts.append((cx+w/2*r*math.cos(a),cy+h/2*r*math.sin(a)))
    s.path(pts,c,INK,.8,True)
    for j in range(4):
        yy=cy-h*.25+j*h*.16
        s.path([(cx-w*.27,yy),(cx-w*.13,yy-3),(cx+w*.04,yy+3),(cx+w*.22,yy)],stroke='#D7E7EC',lw=1.2)
def binder(s,cx,cy,w=56,h=45):
    # Beta strands, connecting loops and a helix form a compact folded binder.
    for j in range(3):
        x=cx-w*.4+j*w*.29;sg=1 if j%2==0 else-1;yt=cy-sg*h*.40;yb=cy+sg*h*.28
        s.path([(x-4,yt),(x+4,yt),(x+4,yb),(x+8,yb),(x,cy+sg*h*.46),(x-8,yb),(x-4,yb)],PURPLE,INK,.55,True)
        if j<2:s.path([(x,yt),(x+8,yt-sg*6),(x+w*.29,yt)],stroke=INK,lw=1)
    pts=[(cx+w*.5+4*math.sin(j*math.tau/12),cy-h*.35+j*h*.7/36)for j in range(37)];s.path(pts,stroke=GOLD,lw=3)
def molecule(s,cx,cy,r=10):
    pts=[(cx+r*math.cos(j*math.pi/3),cy+r*math.sin(j*math.pi/3))for j in range(6)];s.path(pts,stroke='#AB7354',lw=1.5,closed=True)
    for j in [0,2,4]:
        x,y=pts[j];s.line((x,y),(cx+1.7*(x-cx),cy+1.7*(y-cy)),stroke='#AB7354',lw=1.2);s.ellipse(cx+1.7*(x-cx)-2.6,cy+1.7*(y-cy)-2.6,5.2,5.2,['#C26478','#7C9FB6','#AB8DB9'][j//2],INK,.4)
