import argparse, json, urllib.request, zipfile, random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,random_split
from PIL import Image
URL='https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
class Pets(Dataset):
 def __init__(self,root):
  self.r=root; self.img=sorted((root/'PennFudanPed/PNGImages').glob('*.png')); self.mask=root/'PennFudanPed/PedMasks'
 def __len__(self): return len(self.img)
 def __getitem__(self,i):
  x=Image.open(self.img[i]).convert('RGB').resize((128,128)); arr=np.asarray(x,dtype=np.float32)/255
  masks=sorted(self.mask.glob(self.img[i].stem+'*')); m=np.zeros((128,128),np.float32)
  for f in masks:
   ar=np.asarray(Image.open(f).resize((128,128),Image.Resampling.NEAREST)); m=np.maximum(m,(ar>0).astype(np.float32))
  return torch.tensor(arr).permute(2,0,1),torch.tensor(m)[None]
class Block(nn.Module):
 def __init__(self,a,b): super().__init__(); self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())
 def forward(self,x): return self.net(x)
class UNet(nn.Module):
 def __init__(self):
  super().__init__(); self.a=Block(3,16); self.b=Block(16,32); self.c=Block(32,64); self.pool=nn.MaxPool2d(2); self.up1=nn.ConvTranspose2d(64,32,2,2); self.d=Block(64,32); self.up2=nn.ConvTranspose2d(32,16,2,2); self.e=Block(32,16); self.out=nn.Conv2d(16,1,1)
 def forward(self,x):
  a=self.a(x); b=self.b(self.pool(a)); c=self.c(self.pool(b)); d=self.d(torch.cat([self.up1(c),b],1)); return self.out(self.e(torch.cat([self.up2(d),a],1)))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=3); a=p.parse_args(); root=Path(__file__).parent; data=root/'data'; (root/'outputs').mkdir(exist_ok=True); data.mkdir(exist_ok=True)
 z=data/'PennFudanPed.zip'
 if not z.exists(): urllib.request.urlretrieve(URL,z)
 if not (data/'PennFudanPed').exists():
  with zipfile.ZipFile(z) as f:
   for n in f.namelist():
    if '..' in Path(n).parts or n.startswith('/'): raise ValueError('Unsafe archive path')
   f.extractall(data)
 ds=Pets(data); n=len(ds); train_size=int(n*.7); val_size=int(n*.15); test_size=n-train_size-val_size; tr,va,te=random_split(ds,[train_size,val_size,test_size],generator=torch.Generator().manual_seed(42)); device='cuda' if torch.cuda.is_available() else 'cpu'; model=UNet().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); lossfn=nn.BCEWithLogitsLoss()
 def run(part,training=False):
  model.train(training); ls=[]; inter=union=0
  for x,y in DataLoader(part,batch_size=4,shuffle=training):
   x,y=x.to(device),y.to(device); opt.zero_grad(); o=model(x); loss=lossfn(o,y)
   if training: loss.backward(); opt.step()
   ls.append(loss.item()); pred=o.sigmoid()>.5; inter+=int((pred&(y>.5)).sum()); union+=int((pred|(y>.5)).sum())
  return {'loss':round(float(np.mean(ls)),4),'iou':round(inter/max(union,1),4)}
 result={}
 for e in range(a.epochs): result[f'epoch_{e+1}']={'train':run(tr,True),'validation':run(va)}
 result['test']=run(te); torch.save(model.state_dict(),root/'outputs/unet.pt'); (root/'outputs/metrics.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
