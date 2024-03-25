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

batch_size = 32

model_name_or_path ='xlm-roberta-base'
num_layers=12

task = "de"

peft_type = PeftType.LORA
device = "cuda"

lora_dim=8

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=lora_dim, lora_alpha=16, lora_dropout=0.1)

lr = 1e-4

num_epochs = 30

padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("xnli", task)
metric = evaluate.load("xnli", task)


def tokenize_function(examples):
    outputs = tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=None)
    return outputs

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["premise", "hypothesis"],
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


train_split=tokenized_datasets['train'].train_test_split(test_size=0.1,shuffle=False)['test']
print(len(train_split))
#exit()
valid_split=tokenized_datasets["validation"]
train_dataloader = DataLoader(train_split, shuffle=True, collate_fn=collate_fn, batch_size=batch_size,drop_last=True)
eval_dataloader = DataLoader(
    valid_split, shuffle=False, collate_fn=collate_fn, batch_size=batch_size,drop_last=True
)

num_label=3

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_label,return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

optimizer = AdamW(params=model.parameters(), lr=lr)


layers=model.base_model.model.roberta.encoder.layer
param_a=[layer.attention.self.query.a for layer in layers]+[layer.attention.self.value.a for layer in layers]


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
    print('train loss epoch ',epoch+1, epoch_loss / len(train_split))


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


            

