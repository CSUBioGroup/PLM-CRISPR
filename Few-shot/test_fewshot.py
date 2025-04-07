from turtle import forward
import numpy as np
import torch 
import torch.nn as nn 
import torch.utils.data as Data
import data_loader_test_fewshot as dl
import train_fewshot as tt
import os
import matplotlib.pyplot as plt
import scipy.stats as ss
import sys
#import temtrain1 as temtrain

from tensorboardX import SummaryWriter
writer = SummaryWriter(logdir="/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/test_valj/log")
torch.backends.cudnn.benchmark = True
device=0
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
datapath="/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/data_fewshot/test"

protein_dirs = [
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/xCas9.esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/HypaCas9.esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/Sniper-Cas9.esm.pt", 
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/evoCas9.esm.pt"                
              ]
item = os.listdir(datapath)
print(item)
num=0
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
#data_array = np.load('/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/data/test/esp_kim_test.npy')
#print(data_array.shape)  # Check the shape of the array


for sc in item:
    tt=10
    loaded_data = np.genfromtxt(f"/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/traj/trloss", delimiter=',')
    #loaded_data = np.loadtxt(f"/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/OUTPUTloss_12N_0.001/OUTPUT_onlybigtrain-0.3/traj1/trloss")
    nn=len(loaded_data)//tt+1
    sst=[[]for i in range(len(item))]
    net=torch.load(f"/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/model/{tt*(nn-1)}.pt")
    net=net.to(device)
    net.eval()
    cout=0
    for i in item:
        #print("i:",i)
        #valoader1=temtrain.valoader
        data=dl.data_loader(dic=datapath,file=i,filetype=cout,state="prepared")
        #print("1data:",len(data))
        #data=dl.data_loader(protein_dirs,dic=datapath,state="prepared")
        valoader=Data.DataLoader(dataset=data,batch_size=1000,shuffle=True,num_workers=8)
        presave=[]
        effsave=[]
        with torch.no_grad():
            for sgrna,typ, eff in valoader:
                sgrna,typ, eff,protein = sgrna.to(device), typ.to(device), eff.to(device),protein_seqs_tensor.to(device)
                pre=net(sgrna,typ,protein[typ.squeeze(1)],train=False)
                pre=pre.detach().cpu().numpy()
                eff=eff.detach().cpu().numpy()
                #print("eff1",eff1)
                #print("pre1",pre1)
                #print("eff2",eff2)
                #print("pre2",pre2)
                if len(presave)==0 :
                    presave=np.copy(pre)
                    effsave=np.copy(eff)
                else:
                    presave=np.concatenate((presave,pre))
                    effsave=np.concatenate((effsave,eff))
                del sgrna, typ, eff,protein
        cout+=1
       
        effsave= np.array(effsave)
        presave = np.array(presave)   
        np.save("/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/test_valj/presave.npy", presave)
        np.save("/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/test_valj/effsave.npy", effsave)
        np.set_printoptions(threshold=sys.maxsize)
        boxes_pre1=np.load('/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/test_valj/presave.npy',allow_pickle=True)
        np.savetxt('/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/test_valj/presave.txt',boxes_pre1,fmt='%s',newline='\n')
        boxes_eff1=np.load('/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/test_valj/effsave.npy',allow_pickle=True)
        np.savetxt('/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/test_valj/effsave.txt',boxes_eff1,fmt='%s',newline='\n')
        plt.scatter(presave,effsave,s=1)
        spearman_value=ss.spearmanr(effsave.flatten(),presave.flatten())[0]
        writer.add_scalar('OUTPUT_base_Protein/value', spearman_value,num)
        num+=1 
        plt.xlabel("pre")
        plt.ylabel("eff")
        print("spearman:",spearman_value)
        plt.savefig('/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/test_valj/spearman.png')
        plt.show()
        with open(f"/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/Few-shot/output/test_valj/output", 'a') as f:
            f.write(f"len(eff):{len(effsave)}  len(pre):{len(presave)}  Spearman Value: {spearman_value}\n")
writer.close()