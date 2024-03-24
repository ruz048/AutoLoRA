# AutoLoRA

This folder contains the implementation of AutoLoRA in RoBERTa.

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


