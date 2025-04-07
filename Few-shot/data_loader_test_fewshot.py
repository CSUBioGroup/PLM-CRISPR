import numpy as np
import torch 
from torch.utils.data import Dataset
import os

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=0
torch.cuda.set_device(device)

basedir=["A","T","C","G","N"]
transdir={}
cout=0
for i in basedir:
    transdir[i]=cout
    cout+=1

#print(transdir)

class data_loader(Dataset):
    def __init__(self,dic="./data",file=False,filetype=0,state="prepared") -> None:
        super().__init__()
        if not file:
            item=os.listdir(dic)
        else:
            item=[file]
        self.allsgrna=[]
        self.alleff=[]
        self.alltype=[]
        type=0
        for i in item:
            print(i,type if not file else filetype)
            data=np.load(dic+"/"+i,allow_pickle=True)
            print(len(data))
            for psgrna,eff in data:
                line=[]
                try:
                    float(eff)
                except:
                    continue
                if state == "prepared":
                    sgrna = psgrna
                if len(sgrna) == 59 and float(eff) <=1000:
                    for j in range(59):
                        bb=sgrna[j:j+1].upper()
                        line.append(transdir[bb])
                    self.allsgrna.append(line)
                    self.alleff.append(float(eff))
                    self.alltype.append(type)
            type+=1
        self.alleff=np.array(self.alleff).astype(float)
        self.allsgrna=np.array(self.allsgrna).astype(float)
        self.alltype=np.array(self.alltype).astype(float)
        if not file:
            pass
        else:
            self.alltype=np.ones(shape=self.alleff.shape)*filetype
    
    def __len__(self):
        return(len(self.allsgrna))

    def __getitem__(self, index):
        return torch.LongTensor(self.allsgrna[index]),torch.LongTensor([self.alltype[index]]),torch.FloatTensor([self.alleff[index]])