# Parameter-Efficient Fine-Tuning Enhances Adaptation of Single Cell Large Language Model for Cell Type Identification
【论文相关介绍】
## A Quick Overview
![overview](IMG/overview.png)

## Requirements
Download model checkpoint: [scGPT_human](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y). and put it at ./scGPT_human

1.``` git clone https://github.com/laolintou/scPEFT.git ```
2. ```cd scPEFT-main``` and run ```conda env create -f environment.yaml```
3.Activate the conda environment ```conda activate scGPT```

## Get Started
Firstly，enter folder tutorials  ```cd scPEFT-main/tutorials```

### native 
```
python Tutorial_Reference_Mapping.py --data_name "ms"
```
### Command Line Arguments

### Result Output Format
### all finetune
#### train & test
```
python full_finetune.py --data_name "ms" --prompt_type "finetune" --use_prompt False
```
### finetune classifier
#### train & test
```
python finetune_classifier.py --data_name "ms" --use_prompt False
```
### Gene token prompt
#### train & test
```
python gene_token_prompt.py --data_name "ms" --prompt_type "Gene_token_prompt" --use_prompt True
```
### Gene encoder prompt
#### train & test
```
python gene_encoder_prompt.py --data_name "ms" --prompt_type "Gene_encoder_prompt" --use_prompt True
```
### prefix prompt
#### train & test
```
python prefix_prompt.py --data_name "ms" --prompt_type "prefix_prompt" --use_prompt True
```
### LoRA prompt
#### train & test
```
python lora.py --data_name "ms" --prompt_type "LoRA" --use_prompt True
```
### Command Line Arguments
data_name ：dataset name
prompt_type：the type that you add into model
use_prompt：whether use prompt or not
## Data preparation
All data used in this study are publicly available.

The published Zheng68k dataset can be download from [Zheng68k](https://support.10xgenomics.com/single-cell-gene-expression/datasets(SRP073767)).The NSCLC dataset can be download from [NSCLC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE179994).The COVID-19 dataset can be download from [COVID-19](https://figshare.com/articles/dataset/seu_obj_h5ad/16922467/1).The MS dataset can be download from [M.S.]( https://github.com/bowang-lab/scGPT/tree/main/data/)
### Data structure

```
[./data/]
|__[COVID]/
|    |__COVID_test.h5ad
|    |__COVID_train.h5ad
|    |__COVID_val.h5ad
```
## Built With
[pytorch](https://pytorch.org/)
## Citation
