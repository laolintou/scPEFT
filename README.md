# Parameter-Efficient Fine-Tuning Enhances Adaptation of Single Cell Large Language Model for Cell Type Identification
【论文相关介绍】
## A Quick Overview
![overview](IMG/overview.png)

## Requirements
Download model checkpoint: [scGPT_human](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y). and put it at ./scGPT_human

  1.``` git clone https://github.com/laolintou/scPEFT.git ```
  
  2. ```cd scPEFT-main``` and run ```conda env create -f environment.yaml```
     
  3.Activate the conda environment ```conda activate scGPT```
## Installation
## Get Started
### native 
```
python ./tutorials/Tutorial_Reference_Mapping.py
```
### Command Line Arguments

### Result Output Format
### all finetune
#### train & test
```
python ./tutorials/full_finetune.py
```
### finetune classifier
#### train & test
```
python ./tutorials/finetune_classifier.py
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
