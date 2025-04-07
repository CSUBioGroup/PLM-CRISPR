# -*- coding: utf-8 -*- 
import numpy as np
import torch 
import torch.cuda
import torch.nn as nn 
import torch.utils.data as Data
import data_loader_fewshot as dl
import train_fewshot as tt
import os
from tensorboardX import SummaryWriter
writer = SummaryWriter('/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/log')

torch.backends.cudnn.benchmark = True
#torch.manual_seed(3047)
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=0
torch.cuda.set_device(device)
#torch.cuda.device(0)
net=tt.sgrna_net()
net=net.to(device)

lossf = nn.MSELoss()
#lossmr =nn.MarginRankingLoss()
print(net)

basedir=["A","T","C","G","N"]
transdir={}
cout=0
for i in basedir:
    transdir[i]=cout            
    cout+=1

protein_dirs = ["/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/SpCas9.esm.pt",
                 "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/SpCas9-HF1.esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/eSpCas9(1.1).esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/eSpCas9(1.1).esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/SpCas9.esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/SpCas9-HF1.esm.pt",                
               "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/SpCas9.esm.pt",
                 "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/Sniper-Cas9.esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/evoCas9.esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/SpCas9.esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/SpCas9-HF1.esm.pt" 
                ]

train=dl.data_loader(dic="/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/data_fewshot/train",state="prepared")
val=dl.data_loader(dic="/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/data_fewshot/val",state="prepared")

trloader=Data.DataLoader(dataset=train,batch_size=256,shuffle=True,num_workers=8)
valoader=Data.DataLoader(dataset=val ,batch_size=256,shuffle=True,num_workers=8)


opt=torch.optim.Adam(filter(lambda p : p.requires_grad, net.parameters()),lr=0.0001)
she=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=5)   
steps=100
wt1=open("/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/traj/trloss","w+")
wt2=open("/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/traj/valoss","w+")
protein_seqs = []
for file_path in protein_dirs:
    loaded_tensor = torch.load(file_path)
    # Getting the file name from the file path for printing
    file_name = os.path.basename(file_path)
    protein_seq = loaded_tensor['representations'].to(torch.float16)
    protein_seqs.append(protein_seq.numpy())
    # Print file name, size of 'representation', and the loaded tensor content
    #print(file_name, protein_seq.size(), loaded_tensor)
protein_seqs = np.array(protein_seqs)
protein_seqs_tensor = torch.tensor(protein_seqs)


for _ in range(steps+1):
    net.train()
    for sgrna, eff ,typ in trloader:
        sgrna,  typ = sgrna.to(device), typ.to(device)
        eff, protein = eff.to(device),protein_seqs_tensor.to(device)
        #sgrna1 = [''.join([basedir[int(x)] for x in s.tolist()]) for s in sgrna1]
        #sgrna2 = [''.join([basedir[int(x)] for x in s.tolist()]) for s in sgrna2]
        opt.zero_grad()
        #print("sgrna1:",sgrna1.shape,"eff1:",eff1.shape,"typ:",typ.shape)
        #print("1protein shape",protein[typ.squeeze(1)].shape)
        pre=net(sgrna,typ,protein[typ.squeeze(1)])
        loss = lossf(pre, eff)
        loss.backward()
        opt.step()
        she.step()
        torch.cuda.empty_cache()   # 在每次训练迭代结束后释放GPU内存
    writer.add_scalar('fewshot_train/loss', loss.detach().cpu().numpy(), _)
    print(f"Loss: {loss.detach().cpu().numpy()}",  file=wt1, flush=True)
    print(sgrna.shape,typ.shape,pre.shape)
    torch.cuda.empty_cache()

    net.eval()
    with torch.no_grad():
        for sgrna , eff ,typ  in valoader:
            sgrna, typ = sgrna.to(device), typ.to(device)
            eff, protein = eff.to(device),protein_seqs_tensor.to(device)
            #sgrna1 = [''.join([basedir[int(x)] for x in s.tolist()]) for s in sgrna1]
            #sgrna2 = [''.join([basedir[int(x)] for x in s.tolist()]) for s in sgrna2]
            pre=net(sgrna,typ,protein[typ.squeeze(1)],train=False)
            loss = lossf(pre, eff)
        print(f"Loss 1: {loss.detach().cpu().numpy()}", file=wt2,flush=True)
        writer.add_scalar('fewshot_val/loss', loss.detach().cpu().numpy(), _)
    
    torch.cuda.empty_cache()
    if _%10 ==0:
        presave=[]
        effsave=[]
        for sgrna,  eff ,typ  in trloader:
            sgrna, typ = sgrna.to(device), typ.to(device)
            eff,  protein = eff.to(device),protein_seqs_tensor.to(device)
            #sgrna1 = [''.join([basedir[int(x)] for x in s.tolist()]) for s in sgrna1]
            #sgrna2 = [''.join([basedir[int(x)] for x in s.tolist()]) for s in sgrna2]
            pre=net(sgrna,typ,protein[typ.squeeze(1)])        
            pre=pre.detach().cpu().numpy()
            eff=eff.detach().cpu().numpy()
            if len(presave)==0 :
                presave=np.copy(pre)
                effsave=np.copy(eff)
            else:
                presave=np.concatenate((presave,pre))
                effsave=np.concatenate((effsave,eff))
            torch.cuda.empty_cache()
        np.save("/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/out/efftr"+"_"+str(_),effsave)
        np.save("/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/out/pretr"+"_"+str(_),presave)
        
        presave=[]
        effsave=[]
        for sgrna, eff ,typ  in trloader:
            sgrna, typ = sgrna.to(device), typ.to(device)
            eff, protein = eff.to(device), protein_seqs_tensor.to(device)
            pre=net(sgrna,typ,protein[typ.squeeze(1)],train=False)
            pre=pre.detach().cpu().numpy()
            eff=eff.detach().cpu().numpy()
            if len(presave)==0 :
                presave=np.copy(pre)
                effsave=np.copy(eff)
            else:
                presave=np.concatenate((presave,pre))
                effsave=np.concatenate((effsave,eff))
            torch.cuda.empty_cache()
        np.save("/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/out/effval"+"_"+str(_),effsave)
        np.save("/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/out/preval"+"_"+str(_),presave)

        torch.save(net,"/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/model/"+str(_)+".pt")
        writer.close()
        