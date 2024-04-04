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

from bionlp import compute_metrics,label2id,id2label,label_list

import torch
import numpy as np

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem


class Architecture(torch.nn.Module):
    def __init__(self, r_search, n_layer):
        super(Architecture, self).__init__()
        self.alphas=torch.nn.Parameter(torch.zeros(n_layer,r_search*2))

    def forward(self):
        return self.alphas

class Arch(ImplicitProblem):
    def training_step(self, batch):

        alphas = self.forward()

        loss = self.roberta.module(alphas, **batch)  

        return loss


class Roberta(ImplicitProblem):
    def training_step(self, batch):

        alphas = self.arch()
        loss = self.module(alphas, **batch).loss 

        return loss


class NASEngine(Engine):
    @torch.no_grad()
    def validation(self):

        alphas_q,alphas_v=self.arch.module.alphas[:,:lora_dim],self.arch.module.alphas[:,lora_dim:]
        r_list_q=[int(sum(torch.nn.functional.softmax(alpha,dim=-1)>=1/lora_dim)) for alpha in alphas_q]
        r_list_v=[int(sum(torch.nn.functional.softmax(alpha,dim=-1)>=1/lora_dim)) for alpha in alphas_v]

        return {"q": str(r_list_q),'v':str(r_list_v)}
    
model_checkpoint = "roberta-base"
num_layers=12
lora_dim=8
lora_alpha=16
lr = 1e-3
batch_size = 16
num_epochs = 50
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


train_dataset=tokenized_bionlp['train'].train_test_split(test_size=0.3)
train_dataset=train_dataset.remove_columns(["tokens",'tags'])
train_split=train_dataset['train']
search_split=train_dataset['test']
# Instantiate dataloaders.
train_dataloader = DataLoader(train_split, shuffle=True, collate_fn=data_collator, batch_size=batch_size,drop_last=True)
search_dataloader = DataLoader(search_split, shuffle=True, collate_fn=data_collator, batch_size=batch_size,drop_last=True)


model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=11, id2label=id2label, label2id=label2id,
    apply_lora=True,
    lora_alpha=lora_alpha,
    lora_r=lora_dim,
)

lora_parameters =[
        {
            "params": [p for n, p in model.named_parameters() if 'lora_' in n], # if not any(nd in n for nd in no_decay)],
        }]
optimizer = AdamW(params=lora_parameters, lr=lr)
arch_net=Architecture(lora_dim,num_layers)

arch_optimizer = torch.optim.Adam(
    arch_net.parameters(),
    lr=3e-4,
    betas=(0.5, 0.999),
    weight_decay=1e-3,
)
# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

outer_config = Config(retain_graph=True)
unroll_steps=1
inner_config = Config(type="darts", unroll_steps=unroll_steps)

outer = Arch(
    name="arch",
    module=arch_net,
    optimizer=arch_optimizer,
    train_data_loader=search_dataloader,
    config=outer_config,
)
inner = Roberta(
    name="roberta",
    module=model,
    optimizer=optimizer,
    train_data_loader=train_dataloader,
    config=inner_config,
)

problems = [outer, inner]
l2u = {inner: [outer]}
u2l = {outer: [inner]}
dependencies = {"l2u": l2u, "u2l": u2l}


train_portion=1.0
report_freq = 100

num_train = len(train_split)  # 50000
indices = list(range(num_train))
split = int(np.floor(train_portion * num_train))

train_iters = int(
    num_epochs
    * (num_train * train_portion // batch_size + 1)
    * unroll_steps
)
print(train_iters,num_train)

engine_config = EngineConfig(
    valid_step=report_freq * unroll_steps,
    train_iters=train_iters,
    roll_back=True,
)

engine = NASEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()
