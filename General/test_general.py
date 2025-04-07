from turtle import forward
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.data as Data
import data_loader_test_general as dl
import train_general as tt
import os
import matplotlib.pyplot as plt
import scipy.stats as ss
import sys
#import temtrain1 as temtrain

from tensorboardX import SummaryWriter
writer = SummaryWriter(logdir="/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/General/output/test_valj/log")
torch.backends.cudnn.benchmark = True
device = 0
torch.cuda.set_device(device)
datapath = "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/General/data_general/test"

protein_dirs = ["/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/SpCas9.esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/SpCas9-HF1.esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/eSpCas9(1.1).esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/eSpCas9(1.1).esm.pt",                
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/SpCas9.esm.pt",
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/SpCas9-HF1.esm.pt" ,                               
                "/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/data_protein/SpCas9.esm.pt"
                
              ]

item = os.listdir(datapath)
print(item)
num = 0
protein_seqs = []

# Load protein sequences from the given directories
for file_path in protein_dirs:
    loaded_tensor = torch.load(file_path)
    file_name = os.path.basename(file_path)
    protein_seq = loaded_tensor['representations'].to(torch.float16)
    protein_seqs.append(protein_seq.numpy())

protein_seqs = np.array(protein_seqs)
protein_seqs_tensor = torch.tensor(protein_seqs)

def decode_sgrna(sgrna_tensor):
    # 反向映射，从数字到碱基
    basedir = ["A", "T", "C", "G", "N"]
    
    # 将 tensor 中的数字转换为对应的碱基
    sgrna_list = [basedir[int(base)] for base in sgrna_tensor]
    return ''.join(sgrna_list)


for sc in item:
    tt = 10
    loaded_data = np.genfromtxt(f"/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/General/output/traj/trloss", delimiter=',')
    nn = len(loaded_data) // tt + 1
    sst = [[] for i in range(len(item))]
    net = torch.load(f"/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/General/output/model/{tt * (nn - 1)}.pt")
    net = net.to(device)
    net.eval()
    cout = 0

    for i in item:
        # Load data
        data = dl.data_loader(dic=datapath, file=i, filetype=cout, state="prepared")
        valoader = Data.DataLoader(dataset=data, batch_size=1000, shuffle=True, num_workers=8)
        
        presave = []
        effsave = []
        sgrnas = []
        
        # Inference loop
        with torch.no_grad():
            for sgrna, typ, eff in valoader:
                sgrna, typ, eff, protein = sgrna.to(device), typ.to(device), eff.to(device), protein_seqs_tensor.to(device)
                pre = net(sgrna, typ, protein[typ.squeeze(1)], train=False)
                pre = pre.detach().cpu().numpy()
                eff = eff.detach().cpu().numpy()

                # Decode sgrna to actual sequence
                sgrnas.extend([decode_sgrna(seq) for seq in sgrna.cpu().numpy()])
                
                if len(presave) == 0:
                    presave = np.copy(pre)
                    effsave = np.copy(eff)
                else:
                    presave = np.concatenate((presave, pre))
                    effsave = np.concatenate((effsave, eff))
        
        cout += 1

        
        effsave= np.array(effsave)
        presave = np.array(presave)   
        np.save("/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/General/output/test_valj/presave.npy", presave)
        np.save("/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/General/output/test_valj/effsave.npy", effsave)
        np.set_printoptions(threshold=sys.maxsize)
        boxes_pre1=np.load('/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/General/output/test_valj/presave.npy',allow_pickle=True)
        np.savetxt('/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/General/output/test_valj/presave.txt',boxes_pre1,fmt='%s',newline='\n')
        boxes_eff1=np.load('/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/General/output/test_valj/effsave.npy',allow_pickle=True)
        np.savetxt('/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/General/output/test_valj/effsave.txt',boxes_eff1,fmt='%s',newline='\n')
        plt.scatter(presave,effsave,s=1)
        spearman_value=ss.spearmanr(effsave.flatten(),presave.flatten())[0]
        writer.add_scalar('general_test/value', spearman_value,num)
        num+=1 
        plt.xlabel("pre")
        plt.ylabel("eff")
        print("spearman:",spearman_value)
        plt.savefig('/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/General/output/test_valj/spearman.png')
        plt.show()
        with open(f"/public/home/hpc234711034/Project/Cas9-sgRNA/sgRNA_Protein/other/PLM-CRISPR/General/output/test_valj/output", 'a') as f:
            f.write(f"len(eff):{len(effsave)}  len(pre):{len(presave)}  Spearman Value: {spearman_value}\n")

writer.close()
