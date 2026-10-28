"""Native categorical rendering; bold black Times New Roman; no mask interpolation."""
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from prepare import ROOT,sha
FONT=ROOT/'results/B3_fixed_100_epochs_v35_20261023/assets/TimesNewRoman-Bold.ttf'
PAL=np.array([[255,255,255],[136,86,167],[0,158,115],[230,171,2],[0,114,178]],np.uint8)
def isolated(c,mask=None):
    im=np.full((*c.shape,3),255,np.uint8);fg=np.isin(c,[1,2,3,4]);im[fg]=PAL[c[fg]]
    if mask is not None:im[mask&~fg]=[148,148,148]
    return im
def overlay(rgb,c):
    im=rgb.copy();fg=np.isin(c,[1,2,3,4]);im[fg]=np.rint(.25*rgb[fg]+.75*PAL[c[fg]]).astype(np.uint8);return im
def cut(a,r):x,y,x1,y1=r;return a[y:y1,x:x1]
class Canvas:
    def __init__(self,w,h,out):self.im=Image.new('RGB',(w,h),'white');self.d=ImageDraw.Draw(self.im);self.labels=[];self.out=out
    def text(self,x,y,s,size=44,anchor='mt'):
        font=ImageFont.truetype(str(FONT),size);box=self.d.textbbox((x,y),str(s),font=font,anchor=anchor);assert min(box[:2])>=0 and box[2]<=self.im.width and box[3]<=self.im.height,(s,box,self.im.size)
        self.d.text((x,y),str(s),font=font,fill='black',anchor=anchor);self.labels.append(dict(text=str(s),bounds=box))
    def panel(self,a,x,y,w,h,cat=False):
        assert x>=0 and y>=0 and x+w<=self.im.width and y+h<=self.im.height
        self.im.paste(Image.fromarray(a).resize((w,h),Image.Resampling.NEAREST if cat else Image.Resampling.LANCZOS),(x,y));self.d.rectangle((x,y,x+w-1,y+h-1),outline='#DFE4E9',width=2)
    def legend(self,y):
        for j,(s,k) in enumerate([('T1 baseline',4),('Persistent',1),('Newly observed',2),('Tip extension',3)]):
            x=160+j*(self.im.width-240)//4;self.d.rectangle((x,y,x+36,y+36),fill=tuple(PAL[k]));self.text(x+52,y,s,36,'lt')
    def save(self,stem):
        p=self.out/(stem+'.tif');self.im.save(p,compression='tiff_adobe_deflate',dpi=(600,600));prev=self.im.copy();prev.thumbnail((1800,1900));prev.save(self.out/(stem+'_preview.png'))
        return dict(file=str(p.name),sha256=sha(p),dimensions=list(self.im.size),typography=self.labels)
