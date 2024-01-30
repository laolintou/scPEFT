# Parameter-Efficient Fine-Tuning Enhances Adaptation of Single Cell Large Language Model for Cell Type Identification
【论文相关介绍】
## A Quick Overview
![overview](IMG/overview.png)

## Requirements
Download model checkpoint: [scGPT_human]([https://blog.csdn.net/zyz00000000/article/details/82530741?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170659733816777224493685%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=170659733816777224493685&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-82530741-null-null.142^v99^pc_search_result_base9&utm_term=github%E5%9C%A8readme%E4%B8%8A%E4%BC%A0%E7%BD%91%E7%9B%98%E9%93%BE%E6%8E%A5&spm=1018.2226.3001.4187](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y)). and put it at ./scGPT/
environment.yaml
conda env create -f environment.yaml
## Installation
## Get Started
### native 
```
python Tutorial_Reference_Mapping.py
```
### Command Line Arguments

### Result Output Format
### all finetune
#### train & test
```
python full_finetune.py
```
### finetune classifier
#### train & test
```
python finetune_classifier.py
```
### Gene token prompt
#### train & test
```
python ./tutorials/gene_token_prompt.py
```
### Gene encoder prompt
#### train & test
```
python ./tutorials/gene_encoder_prompt.py
```
### prefix prompt
#### train & test
```
python ./tutorials/prefix_prompt.py
```
### LoRA prompt
#### train & test
```
python ./tutorials/lora.py
```
### Command Line Arguments
## Data preparation
### Data structure

## Built With
[pytorch](https://pytorch.org/)
## Citation
