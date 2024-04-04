from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from bionlp import compute_metrics,label2id,id2label,label_list
from peft.utils import globals

import torch
import numpy as np

model_checkpoint = "roberta-base"
num_layers=12
lora_dim=8
lr = 1e-3
batch_size = 64
num_epochs = 30
device='cuda'

bionlp = load_dataset("tner/bionlp2004")


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_bionlp = bionlp.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt",)

if globals.search:
    train_dataset=tokenized_bionlp['train'].train_test_split(test_size=0.3)
    train_dataset=train_dataset.remove_columns(["tokens",'tags'])
    train_split=train_dataset['train']
    search_split=train_dataset['test']
    # Instantiate dataloaders.
    train_dataloader = DataLoader(train_split, shuffle=True, collate_fn=data_collator, batch_size=batch_size,drop_last=True)
    search_dataloader = DataLoader(search_split, shuffle=True, collate_fn=data_collator, batch_size=batch_size,drop_last=True)
else:
    train_split=tokenized_bionlp['train'].remove_columns(["tokens",'tags'])
    train_dataloader = DataLoader(train_split, shuffle=True, collate_fn=data_collator, batch_size=batch_size,drop_last=True)
    eval_dataloader = DataLoader(
        tokenized_bionlp["test"].remove_columns(["tokens",'tags']), shuffle=False, collate_fn=data_collator, batch_size=batch_size,drop_last=True
    )

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=11, id2label=id2label, label2id=label2id
)

peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, inference_mode=False, r=lora_dim, lora_alpha=16, lora_dropout=0.1, bias="all"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


optimizer = AdamW(params=model.parameters(), lr=lr)


if model_checkpoint == "roberta-base" or model_checkpoint == "roberta-large":
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

            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            epoch_loss_search += batch_size * loss.item()
            loss.backward()
            optimizer_a.step()
            optimizer_a.zero_grad()

    print('train loss epoch ',epoch+1, epoch_loss / len(train_split))

    if globals.search:
        print('train loss search epoch ',epoch+1, epoch_loss_search / len(train_split))
        if model_checkpoint == "roberta-base" or model_checkpoint == "roberta-large":
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
        pred,ref=[],[]

        for step, batch in enumerate((eval_dataloader)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            pred.append(predictions.cpu().detach().numpy())
            ref.append(batch['labels'].cpu().detach().numpy())

        eval_metric = compute_metrics(zip(pred,ref))
        print(f"epoch {epoch}:", eval_metric)
