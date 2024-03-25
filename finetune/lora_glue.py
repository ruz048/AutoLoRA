import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
from peft.utils import globals
import evaluate
from datasets import load_dataset,concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm

batch_size = 16
#model_name_or_path = "roberta-base"
#model_name_or_path = "roberta-large"
model_name_or_path = "microsoft/deberta-v2-xxlarge"
#model_name_or_path ='xlm-roberta-base'
#model_name_or_path = "microsoft/deberta-v2-xxlarge"
num_layers=96

#task = "mrpc"
#task = 'qqp'
#task='qnli'
task='sst2'
#task='rte'
#task='cola'
#task='mnli'
#task='stsb'
peft_type = PeftType.LORA
device = "cuda"

lora_dim=8
peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=lora_dim, lora_alpha=16, lora_dropout=0.1)
'''
if model_name_or_path == "roberta-base" or model_name_or_path == "roberta-large" or model_name_or_path == "xlm-roberta-base":
    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=lora_dim, lora_alpha=16, lora_dropout=0.1)
else:
    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=lora_dim, lora_alpha=16, 
                           lora_dropout=0.1,target_modules=['query_proj','value_proj'])
'''

lr = 1e-4

if globals.search:
    num_epochs = 50
else:
    if task=='cola':
        num_epochs = 80
    if task=='qqp' or task=='mnli':
        num_epochs = 10
    else:
        num_epochs = 30

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("glue", task)
metric = evaluate.load("glue", task)


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    if task=='mrpc' or task=='rte' or task=='stsb':
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
    elif task=='qqp':
        outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=None)
    elif task=='qnli':
        outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=None)
    elif task=='sst2' or task=='cola':
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=None)
    elif task=='mnli':
        outputs = tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=None)
    return outputs

if task=='mrpc' or task=='rte' or task=='stsb':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )
elif task=='qqp':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "question1", "question2"],
    )
elif task=='qnli':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "question", "sentence"],
    )
elif task=='mnli':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "premise", "hypothesis"],
    )
elif task=='sst2' or task=='cola':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx","sentence"],
    )
# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

if globals.search:
    train_dataset=tokenized_datasets['train'].train_test_split(test_size=0.3)
    train_split=train_dataset['train']
    search_split=train_dataset['test']
    # Instantiate dataloaders.
    train_dataloader = DataLoader(train_split, shuffle=True, collate_fn=collate_fn, batch_size=batch_size,drop_last=True)
    search_dataloader = DataLoader(search_split, shuffle=True, collate_fn=collate_fn, batch_size=batch_size,drop_last=True)
else:
    train_split=tokenized_datasets['train']
    if task=='mnli':
        #valid_split=tokenized_datasets["validation_matched"]
        valid_split=concatenate_datasets([tokenized_datasets["validation_matched"], tokenized_datasets["validation_mismatched"]])
    else:
        valid_split=tokenized_datasets["validation"]
    train_dataloader = DataLoader(train_split, shuffle=True, collate_fn=collate_fn, batch_size=batch_size,drop_last=True)
    eval_dataloader = DataLoader(
        valid_split, shuffle=False, collate_fn=collate_fn, batch_size=batch_size,drop_last=True
    )

if task=='mnli':num_label=3
elif task=='stsb':num_label=1
else:num_label=2

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_label,return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

optimizer = AdamW(params=model.parameters(), lr=lr)

if model_name_or_path == "roberta-base" or model_name_or_path == "roberta-large" or model_name_or_path=='xlm-roberta-base':
    layers=model.base_model.model.roberta.encoder.layer
    param_a=[layer.attention.self.query.a for layer in layers]+[layer.attention.self.value.a for layer in layers]
else:
    layers=model.base_model.model.deberta.encoder.layer
    param_a=[layer.attention.self.query_proj.a for layer in layers]+[layer.attention.self.value_proj.a for layer in layers]

optimizer_a = torch.optim.Adam(param_a,lr=0.001)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model.to(device)
for epoch in range(num_epochs):
    model.train()
    #for step, batch in enumerate(tqdm(train_dataloader)):
    epoch_loss,epoch_loss_search=0,0
    for step, batch in enumerate((train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        epoch_loss += batch_size * loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        if globals.search:
            batch = next(iter(search_dataloader))
            #print(batch)
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            epoch_loss_search += batch_size * loss.item()
            loss.backward()
            optimizer_a.step()
            optimizer_a.zero_grad()
        #print(model.base_model.model.roberta.encoder.layer[0].attention.self.query.a)
    print('train loss epoch ',epoch+1, epoch_loss / len(train_split))

    if globals.search:
        print('train loss search epoch ',epoch+1, epoch_loss_search / len(train_split))
        if model_name_or_path == "roberta-base" or model_name_or_path == "roberta-large":
            query_rlist=[int(sum(torch.nn.functional.softmax(layer.attention.self.query.a,dim=0)>=1/lora_dim)) for layer in layers]
            value_rlist=[int(sum(torch.nn.functional.softmax(layer.attention.self.value.a,dim=0)>=1/lora_dim)) for layer in layers]
        else:
            query_rlist=[int(sum(torch.nn.functional.softmax(layer.attention.self.query_proj.a,dim=0)>=1/lora_dim)) for layer in layers]
            value_rlist=[int(sum(torch.nn.functional.softmax(layer.attention.self.value_proj.a,dim=0)>=1/lora_dim)) for layer in layers]
        rlist=[]
        for i in range(num_layers):
            rlist.append(query_rlist[i])
            rlist.append(value_rlist[i])
        
        print(rlist)
    else:
        model.eval()
        #for step, batch in enumerate(tqdm(eval_dataloader)):
        for step, batch in enumerate((eval_dataloader)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            if task!='stsb':
                predictions = outputs.logits.argmax(dim=-1)
            else:
                predictions = outputs.logits
            
            predictions, references = predictions, batch["labels"].float()
            #print(predictions,references)
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print(f"epoch {epoch}:", eval_metric)


            

