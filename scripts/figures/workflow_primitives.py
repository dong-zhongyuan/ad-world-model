# -*- coding: utf-8 -*-
import json, math, csv, pathlib, html
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'output'; INK='#263746'; BLUE='#BBD9EA'; CORAL='#E9A7A5'; GREEN='#BFD9BF'; GOLD='#F2D99D'; PURPLE='#CFC5E2'; GRAY='#EEF1F3'
scenes=[]
class Scene:
 def __init__(self,n,w,h,refs): self.n=n;self.w=w;self.h=h;self.refs=refs;self.e=[];self.group='frame';self.text(4,3,18,16,'A',13,True,'left',color='#000000')
 def add(self,t,**kw): self.e.append(dict(id=f'F{self.n}_{len(self.e)+1:04d}',group=self.group,type=t,**kw))
 def rect(self,x,y,w,h,fill='none',stroke=INK,lw=2.5):self.add('rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,lw=lw)
 def ellipse(self,x,y,w,h,fill='none',stroke=INK,lw=2.5):self.add('ellipse',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,lw=lw)
 def path(self,pts,fill='none',stroke=INK,lw=2.5,closed=False):self.add('path',pts=pts,fill=fill,stroke=stroke,lw=lw,closed=closed)
 def line(self,*pts,stroke=INK,lw=3):self.path(list(pts),stroke=stroke,lw=lw)
 def arrow(self,*pts,stroke=INK,lw=3.3,head=12):
  self.line(*pts,stroke=stroke,lw=lw);x,y=pts[-1];px,py=pts[-2];a=math.atan2(y-py,x-px);self.path([(x,y),(x-head*math.cos(a)+head*.45*math.sin(a),y-head*math.sin(a)-head*.45*math.cos(a)),(x-head*math.cos(a)-head*.45*math.sin(a),y-head*math.sin(a)+head*.45*math.cos(a))],fill=stroke,stroke=stroke,lw=.4,closed=True)
 def text(self,x,y,w,h,s,size=28,bold=False,align='center',color=INK):self.add('text',x=x,y=y,w=w,h=h,text=s,size=size,bold=bold,align=align,color=color)
 def dashbox(self,x,y,w,h,color=INK):
  for xx in range(int(x),int(x+w),19):self.line((xx,y),(min(xx+10,x+w),y),stroke=color,lw=2);self.line((xx,y+h),(min(xx+10,x+w),y+h),stroke=color,lw=2)
  for yy in range(int(y),int(y+h),19):self.line((x,yy),(x,min(yy+10,y+h)),stroke=color,lw=2);self.line((x+w,yy),(x+w,min(yy+10,y+h)),stroke=color,lw=2)
 def matrix(self,x,y,cols=6,rows=4,cell=15,palette=None):
  palette=palette or [BLUE,'#EDF5FA','#87B4CE'];
  for r in range(rows):
   for c in range(cols):self.rect(x+c*cell,y+r*cell,cell-2,cell-2,palette[(r*3+c*7+r*c)%len(palette)],'none',0)
  self.line((x-5,y),(x-10,y),(x-10,y+rows*cell-2),(x-5,y+rows*cell-2),lw=2)
  self.line((x+cols*cell+1,y),(x+cols*cell+6,y),(x+cols*cell+6,y+rows*cell-2),(x+cols*cell+1,y+rows*cell-2),lw=2)
 def cells(self,x,y,scale=1,kind='blood'):
  pts=[(0,26),(31,2),(64,16),(16,66),(57,59),(87,47)]
  for i,(dx,dy) in enumerate(pts):
   c=[CORAL,BLUE,GREEN,PURPLE][i%4] if kind=='mix' else (CORAL if kind=='blood' else PURPLE)
   self.ellipse(x+dx*scale,y+dy*scale,26*scale,26*scale,c,INK,1.6*scale)
   self.ellipse(x+(dx+8)*scale,y+(dy+8)*scale,10*scale,10*scale,'#F7EEEE' if kind=='blood' else '#9F8BB9','none',0)
 def brain(self,x,y,scale=1):
  pts=[(10,38),(8,27),(16,13),(31,9),(43,3),(57,8),(70,5),(85,17),(95,25),(95,40),(105,51),(98,66),(91,74),(76,76),(66,87),(51,83),(44,70),(29,70),(18,62),(10,51)]
  smooth=[]
  for i in range(len(pts)):
   p0,p1,p2,p3=[pts[j%len(pts)] for j in (i-1,i,i+1,i+2)]
   for j in range(8):
    t=j/8
    a,b=[.5*((2*p1[k])+(-p0[k]+p2[k])*t+(2*p0[k]-5*p1[k]+4*p2[k]-p3[k])*t*t+(-p0[k]+3*p1[k]-3*p2[k]+p3[k])*t*t*t)for k in (0,1)]
    smooth.append((x+a*scale,y+b*scale))
  self.path(smooth,PURPLE,INK,2.2*scale,True)
  for ps in [[(31,13),(30,28),(42,36),(40,53)],[(59,13),(51,27),(63,40),(59,60),(66,74)],[(80,22),(70,30),(81,44),(79,59)],[(19,42),(30,43),(28,57)]]:self.line(*[(x+a*scale,y+b*scale)for a,b in ps],stroke='#8F7AA9',lw=2*scale)
 def paired(self,x,y,scale=1):
  self.ellipse(x+18*scale,y+15*scale,64*scale,64*scale,GREEN,INK,2)
  self.ellipse(x+39*scale,y+36*scale,22*scale,22*scale,'#8DB58E','none',0)
  for a in range(0,360,45):
   rad=a*math.pi/180;cx=x+50*scale+35*scale*math.cos(rad);cy=y+47*scale+35*scale*math.sin(rad)
   self.line((cx,cy),(cx+12*scale*math.cos(rad),cy+12*scale*math.sin(rad)),stroke='#677C3F',lw=2)
   self.ellipse(cx+12*scale*math.cos(rad)-3,cy+12*scale*math.sin(rad)-3,6,6,GOLD,INK,1)
 def lock(self,x,y,s=1):
  self.ellipse(x+7*s,y,18*s,24*s,'none',INK,2*s);self.rect(x,y+14*s,32*s,25*s,'#FFFFFF',INK,2*s);self.ellipse(x+13*s,y+22*s,6*s,6*s,INK,'none',0)
 def stack(self,x,y,w=100,h=60,color=BLUE):
  for d in [14,7,0]:self.rect(x+d,y-d,w,h,color,INK,2)
 def spatial(self,x,y,s=1):
  self.path([(x,y+25*s),(x+30*s,y),(x+90*s,y+8*s),(x+114*s,y+45*s),(x+90*s,y+75*s),(x+35*s,y+88*s),(x+2*s,y+64*s)],'#F5E3E2',INK,2,True)
  for r in range(5):
   for c in range(7):self.ellipse(x+(16+c*12+r%2*5)*s,y+(18+r*12)*s,5*s,5*s,[CORAL,PURPLE,BLUE][(c+r)%3],'none',0)
 def database(self,x,y,s=1,color=GOLD):
  self.rect(x,y+12*s,96*s,68*s,color,INK,2);self.ellipse(x,y+64*s,96*s,25*s,color,INK,2);self.ellipse(x,y,96*s,25*s,color,INK,2)
  self.line((x,y+40*s),(x+96*s,y+40*s),lw=1.6)
 def capsule(self,x,y,s=1):
  self.rect(x+12*s,y,70*s,30*s,'#FFFFFF',INK,2);self.ellipse(x,y,28*s,30*s,CORAL,INK,2);self.rect(x+14*s,y+1*s,34*s,28*s,CORAL,'none',0);self.ellipse(x+68*s,y,28*s,30*s,'#FFFFFF',INK,2);self.rect(x+49*s,y+1*s,32*s,28*s,'#FFFFFF','none',0);self.line((x+48*s,y),(x+48*s,y+30*s),lw=2)
 def module(self,x,y,w,h,label,color=BLUE,size=29):self.rect(x,y,w,h,color,INK,2.7);self.text(x+8,y+8,w-16,h-16,label,size,True)
 def finish(self): scenes.append(dict(figure=self.n,width=self.w,height=self.h,reference_panels=self.refs,elements=self.e))
# Additional diagram elements: biological illustrations remain individual image layers;
# their detailed geometry is editable in the matching Asset_*.blend.
def asset(self,key,x,y,w,h):self.add('image',asset=key,x=x,y=y,w=w,h=h)
def block(self,x,y,w,h,label,color=BLUE,size=30,depth=12):
 self.path([(x,y),(x+depth,y-depth),(x+w+depth,y-depth),(x+w,y)],fill='#F1F5F6',closed=True,lw=2)
 self.path([(x+w,y),(x+w+depth,y-depth),(x+w+depth,y+h-depth),(x+w,y+h)],fill='#A5B5C2',closed=True,lw=2)
 self.module(x,y,w,h,label,color,size)
def snow(self,x,y,size=18):
 for a in range(0,360,60):
  t=math.radians(a);end=(x+size*math.cos(t),y+size*math.sin(t));self.line((x,y),end,stroke='#3477AA',lw=2.7)
  for sg in [-1,1]:
   xx=x+size*.61*math.cos(t);yy=y+size*.61*math.sin(t);self.line((xx,yy),(xx+size*.37*math.cos(t+sg*.85),yy+size*.37*math.sin(t+sg*.85)),stroke='#3477AA',lw=2)
def flame(self,x,y,s=1):self.path([(x,y+15*s),(x-11*s,y+5*s),(x-8*s,y-8*s),(x-2*s,y-2*s),(x+3*s,y-20*s),(x+13*s,y),(x+10*s,y+12*s)],fill='#E57759',stroke='#BE583C',closed=True,lw=1.5)
def groupcloud(self,x,y,w,h,colors,shapes=False):
 # Dashed ellipsoid copied in structure from reference 3A/4A; each token has semantic color.
 for j in range(28):
  a=j*math.tau/28;b=a+.12;pts=[(x+w/2+w/2*math.cos(t),y+h/2+h/2*math.sin(t))for t in [a,(a+b)/2,b]];self.path(pts,stroke='#294864',lw=2.5)
 pts=[(.24,.33),(.54,.22),(.76,.45),(.38,.57),(.64,.71),(.19,.75),(.77,.2)]
 for j,(u,v) in enumerate(pts):
  xx=x+w*u;yy=y+h*v;c=colors[j%len(colors)]
  if shapes and j%2:self.path([(xx,yy-10),(xx-10,yy+8),(xx+10,yy+8)],c,INK,2,True)
  else:self.ellipse(xx-9,yy-9,18,18,c,INK,2)
def network(self,x,y,w,h,color=BLUE):
 for a in range(2):
  for j in range(4):
   for k in range(4):self.line((x+a*w/2,y+j*h/3),(x+(a+1)*w/2,y+k*h/3),stroke='#B8C8D2',lw=.8)
 for a in range(3):
  for j in range(4):self.ellipse(x+a*w/2-5,y+j*h/3-5,10,10,color,INK,1.3)
def chip(self,x,y,w,label,color=BLUE,size=25):self.rect(x,y,w,43,color,INK,1.8);self.text(x+3,y+3,w-6,37,label,size,True)
def barcode(self,x,y,w,h):
 xx=x
 for i in range(29):
  bw=[1,3,2,1,4][i%5]*w/97;self.rect(xx,y,bw,h,INK,'none',0);xx+=bw+w/97
Scene.asset=asset;Scene.block=block;Scene.snow=snow;Scene.flame=flame;Scene.cloud=groupcloud;Scene.network=network;Scene.chip=chip;Scene.barcode=barcode
