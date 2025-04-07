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



'''  
if __name__=="__main__": 
    




  
    train=data_loader(dic="/home/houyalin123/Project/Cas9-sgRNA/sgRNA_Protein/data/train",state="prepared",sample_ratio=1)
    traloader=Data.DataLoader(dataset=train,batch_size=256,shuffle=True,num_workers=4)
    torch.save(traloader, '/home/houyalin123/Project/Cas9-sgRNA/sgRNA_Protein/data/traloader.pt')
    test=data_loader(dic="/home/houyalin123/Project/Cas9-sgRNA/sgRNA_Protein/data/test",state="prepared",sample_ratio=1)
    testloader=Data.DataLoader(dataset=test,batch_size=1000,shuffle=True,num_workers=4)
    torch.save(testloader, '/home/houyalin123/Project/Cas9-sgRNA/sgRNA_Protein/data/testloader.pt')


traloader = torch.load('traloader.pt')
traloader_iter = iter(traloader)
for inputs, labels in traloader_iter:
    # do something

    for _ in range(10):
        for sgrna1, sgrna2,  eff1, eff2 ,typ in traloader:
            sgrna1, sgrna2, typ = sgrna1.to(device), sgrna2.to(device), typ.to(device)
            eff1, eff2 = eff1.to(device), eff2.to(device)
            #sgrna1 = [''.join([basedir[int(x)] for x in s.tolist()]) for s in sgrna1]
            #sgrna2 = [''.join([basedir[int(x)] for x in s.tolist()]) for s in sgrna2]
            # Now sgrna1 and sgrna2 are lists of strings
            # You can do whatever you want with them
            print("sgrna1:",sgrna1.shape,"eff1:",eff1.shape,"typ:",typ.shape)
            #print("当前批次类型统计:", torch.unique(typ, return_counts=True))
            #print("sgrna1",sgrna1.shape,"eff1",eff1.shape,"sgrna2",sgrna1.shape,"eff2",eff1.shape,"typ",typ[1])

        #print("1protein shape",protein[typ.squeeze(1)].shape)

    with open("/home/houyalin123/Project/Cas9-sgRNA/sgRNA_Protein/other/testoutput/printoutput.txt", "w") as file:
        for i in range(20):
            sgrna1, sgrna2, typ, eff1, eff2 = test[i]
            output_line = f"Sequence1 length: {len(sgrna1)}, Type: {typ.item()},Efficiency1: {eff1.item()},Sequence2 length: {len(sgrna1)}, Efficiency2: {eff2.item()},protein length: {len(protein)}\n"
            file.write(output_line)
 

esp_kim_test.npy 0
1368
HF_kim_test.npy 1
1368
HF_wang_test.npy 2
11889
esp_wang_test.npy 3
11889
Hypa_test.npy 4
262
evo_test.npy 5
303
WT_kim_test.npy 6
1368
WT_wang_test.npy 7
11889
WT_xiang_test.npy 8
2135
sniper_test.npy 9
306
xcas9_test.npy 10
304
 '''