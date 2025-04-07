import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.data as Data
import os
import pandas as pd

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=0
torch.cuda.set_device(device)

# 定义碱基映射字典
basedir=["A","T","C","G","N"]
transdir={}
cout=0
for i in basedir:
    transdir[i]=cout
    cout+=1
# 打印碱基映射字典
#print(transdir)

# 定义数据加载器类
class data_loader(Dataset):
    def __init__(self,dic="./data",file=False,filetype=0,state="prepared") -> None:#未指定文件
        super().__init__()
        # 如果未指定文件，则加载目录下所有文件
        if not file:
            item=os.listdir(dic)
            #print("filetypeitem:",item)
        else:
            item=[file]
            #print("item:",item)
        self.allsgrna=[]
        self.alleff=[]
        self.alltype=[]
        type=0
        for i in item:
            print(i,type if not file else filetype)
            #if not file:
                #print(i, type)
            #else:
                #print(i, filetype)
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
                    self.alleff.append(eff)
                    self.alltype.append(type)
            type+=1
            
        #print ("len(sgrna1):",len(self.allsgrna1),"len(sgrna2):",len(self.allsgrna2))
        #print ("len(eff1):",len(self.alleff1),"len(eff2):",len(self.alleff2))
        self.allsgrna=np.array(self.allsgrna).astype(np.float16)
        self.alleff=np.array(self.alleff).astype(np.float16)
        self.alltype=np.array(self.alltype).astype(int)
        #print("alltype",self.alltype)
        print("allsgrna",len(self.allsgrna),"alleff",len(self.alleff),"alltyp",len(self.alltype))
        # 如果指定了文件类型，则将所有数据类型设为该类型
        if not file:
            pass
        else:
            self.alltype=np.ones(shape=self.alleff1.shape)*filetype
    
    def __len__(self):
        return len(self.allsgrna)

    def __getitem__(self, index):
        
        # 返回sgRNA序列、类型和效能值、protein编码的张量
        return (torch.LongTensor(self.allsgrna[index]),
        torch.FloatTensor([self.alleff[index]]),
        torch.LongTensor([self.alltype[index]]),
        )