# AutoLoRA

This folder contains the implementation of AutoLoRA ("AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation
Based on Meta Learning") in RoBERTa.

Paper Link: https://arxiv.org/pdf/2403.09113.pdf

## Steps to reproduce our results

### Install the pre-requisites

```console
pip install -r requirements.txt
```
(Note that the source code of transformer library and Betty library AutoLoRA uses is different from that in the main branch)
### Start the experiments 

Search for the optimal rank of RoBERTa-base model on CoLA dataset

```console
sh glue_search.sh
```

After obtaining optimal rank list, define the optimal ranks r_list in finetune/peft/utils/globals.py For example:

```console
r_list=[3, 2, 4, 4, 5, 3, 3, 4, 5, 3, 5, 4, 3, 3, 5, 4, 2, 5, 4, 4, 3, 4, 3, 3] 
```

Finetuning the RoBERTa-base model with optimal LoRA rank:

```console
python lora_glue.py
```

