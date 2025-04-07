# **PLM-CRISPR**

Protein sequence feature extraction using the Protein Big Language model was fused with sgRNA sequence features for sgRNA activity prediction of different Cas9 variants.

This repository contains the code for the Predictive Model for Multiple Cas9 Variant sgrna Activity (PLM-CRISPR), which can be used to predict the sgRNA activity of different Cas9 variants under three data scenarios: general, few-shot and zero-shot.

Before you begin, you need to create the environment needed for your project.

## Create Environment 

First, create the required environment and activate it.

```python
cd PLM-CRISPR
conda create -n PLM-CRISPR python=3.8.0
conda activate PLM-CRISPR
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

Then, follow requirements.txt to create the environment.

```
pandas==2.0.3
matplotlib==3.7.5
tensorboardX==2.6.2.2
numpy==1.23.5
scipy==1.10.1
```

## Get Started

### Protein Language Model Download(Optional)

The protein language model used by PLM-CRISPR is esm-2, which is available in [GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#available-models).Then follow it to download the ESM model and configure the environment, and finally enter the protein variant sequences for embedding.

More simply, you can use the features we have extracted for each protein variant, saved in the data_protein directory.

### Training

You can train models in a very simple way. For different data scenarios, just train the python file with the appropriate files.

```
python temtrain_fewshot.py
python temtrain_general.py
python temtrain_zeroshot.py
```

To change model hyperparameters, you can modify them in the corresponding files.The model hyperparameters are as follows:

```python
protein_dirs: the folder where the protein embedding is located
data_dir: the folder where the sgRNA data is located.
log_dir: the location where the log file is located.
model_dir: the location where the model is located.
seed: random Seed
device: specified gpuid
state: default prepared
shuffle: default True
num_workers:  number of worker processes used when loading data
dataset: select training dataset, test dataset or validation dataset.
batch_size: batch size
lr: learning rate
embed_size: dimension of protein feature embedding
kernel_sizes: multi-scale convolutional kernel settings used by TextCNN
dropout: dropout
```

After training, you will find the trained model saved in the `model` folder. You can use these model for making predictions.

### Prediction

For prediction, you can place the data file and model file in the working directory and run the following command to get the  prediction results:

```python
python test_fewshot.py
python test_general.py
python test_zeroshot.py
```

## Citation

Yalin Hou #, Yiming Li #, Chengqian Lu, Ruiqing Zheng, Fei Guo, Min Li, Min Zeng*, "Leveraging protein language models for cross-variant CRISPR/Cas9 sgRNA activity prediction".
