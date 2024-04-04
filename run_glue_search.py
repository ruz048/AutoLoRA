#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import torch
from torch.utils.data import DataLoader

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from utils import task_to_keys,ModelArguments,DataTrainingArguments

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

logger = logging.getLogger(__name__)

class Architecture(torch.nn.Module):
    def __init__(self, r_search, n_layer):
        super(Architecture, self).__init__()
        self.alphas=torch.nn.Parameter(torch.ones(n_layer,r_search*2))

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


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

class NASEngine(Engine):
    @torch.no_grad()
    def validation(self):

        alphas_q,alphas_v=self.arch.module.alphas[:,:model_args.lora_r],self.arch.module.alphas[:,model_args.lora_r:]

        r_list_q=[int(sum(alpha/torch.sum(alpha)>=1/model_args.lora_r)) for alpha in alphas_q]
        r_list_v=[int(sum(alpha/torch.sum(alpha)>=1/model_args.lora_r)) for alpha in alphas_v]
        r_list=[]
        for q,v in zip(r_list_q,r_list_v):
            r_list.append(q)
            r_list.append(v)

        return {"q": str(r_list_q),'v':str(r_list_v),'r':str(r_list)}

training_args.use_deterministic_algorithms=False
torch.use_deterministic_algorithms(training_args.use_deterministic_algorithms)
logger.info("use_deterministic_algorithms: " + str(torch.are_deterministic_algorithms_enabled()))

# Detecting last checkpoint.
last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
logger.info(f"Training/evaluation parameters {training_args}")

# Set seed before initializing model.
set_seed(training_args.seed)

# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
# or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
#
# For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
# sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
# label if at least two columns are provided.
#
# If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
# single column. You can easily tweak this behavior (see below)
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
if data_args.task_name is not None:
    # Downloading and loading a dataset from the hub.
    datasets = load_dataset("glue", data_args.task_name)
else:
    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset("csv", data_files=data_files)
    else:
        # Loading a dataset from local json files
        datasets = load_dataset("json", data_files=data_files)
# See more about loading any type of standard or custom dataset at
# https://huggingface.co/docs/datasets/loading_datasets.html.

# Labels
if data_args.task_name is not None:
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
else:
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

# Load pretrained model and tokenizer
#
# In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
training_args.cls_dropout=0.1
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
    cls_dropout=training_args.cls_dropout,
    apply_lora=model_args.apply_lora,
    lora_alpha=model_args.lora_alpha,
    lora_r=model_args.lora_r,
    apply_adapter=model_args.apply_adapter,
    adapter_type=model_args.adapter_type,
    adapter_size=model_args.adapter_size,
    reg_loss_wgt=model_args.reg_loss_wgt,
    masking_prob=model_args.masking_prob,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

trainable_params = []
if model_args.apply_lora:
    if model_args.lora_path is not None:
        lora_state_dict = torch.load(model_args.lora_path)
        logger.info(f"Apply LoRA state dict from {model_args.lora_path}.")
        logger.info(lora_state_dict.keys())
        model.load_state_dict(lora_state_dict, strict=False)
    trainable_params.append('lora')

if len(trainable_params) > 0:
    for name, param in model.named_parameters():
        if name.startswith('deberta') or name.startswith('roberta'):
            param.requires_grad = False
            for trainable_param in trainable_params:
                if trainable_param in name:
                    param.requires_grad = True
                    break
        else:
            param.requires_grad = True

# Preprocessing the datasets
if data_args.task_name is not None:
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
else:
    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

# Padding strategy
if data_args.pad_to_max_length:
    padding = "max_length"
else:
    # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    padding = False

# Some models have set the order of the labels to use, so let's make sure we do use it.
label_to_id = None
if (
    model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    and data_args.task_name is not None
    and not is_regression
):
    # Some have all caps in their config, some don't.
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
        label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    else:
        logger.warn(
            "Your model seems to have been trained with labels, but they don't match the dataset: ",
            f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
            "\nIgnoring the model labels as a result.",
        )
elif data_args.task_name is None and not is_regression:
    label_to_id = {v: i for i, v in enumerate(label_list)}

if data_args.max_seq_length > tokenizer.model_max_length:
    logger.warn(
        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result

if sentence2_key:
    datasets = datasets.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key, sentence2_key],
                        load_from_cache_file=not data_args.overwrite_cache)
else:
    datasets = datasets.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key],
                        load_from_cache_file=not data_args.overwrite_cache)

if training_args.do_train:
    if "train" not in datasets:
        raise ValueError("--do_train requires a train dataset")
    dataset = datasets["train"].train_test_split(test_size=0.3)
    train_dataset,eval_dataset=dataset['train'],dataset['test']
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
    if "test" not in datasets and "test_matched" not in datasets:
        raise ValueError("--do_predict requires a test dataset")
    test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
    if data_args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(data_args.max_test_samples))

train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator,
                                batch_size=training_args.per_device_train_batch_size,drop_last=True)
eval_dataloader = DataLoader(eval_dataset, shuffle=True, collate_fn=default_data_collator,
                                batch_size=training_args.per_device_train_batch_size,drop_last=True)

unroll_steps=1
train_portion=1.0
report_freq = 100

outer_config = Config(retain_graph=True)
inner_config = Config(type="darts", unroll_steps=unroll_steps)

num_train = len(train_dataset)  # 50000
indices = list(range(num_train))
split = int(np.floor(train_portion * num_train))

train_iters = int(
    training_args.num_train_epochs
    * (num_train * train_portion // training_args.per_device_train_batch_size + 1)
    * unroll_steps
)

engine_config = EngineConfig(
    valid_step=report_freq * unroll_steps,
    train_iters=train_iters,
    roll_back=True,
)

lora_parameters =[
        {
            "params": [p for n, p in model.named_parameters() if 'lora_' in n], # if not any(nd in n for nd in no_decay)],
        }]

optimizer = torch.optim.Adam(lora_parameters, lr=3e-4)

arch_net=Architecture(model_args.lora_r,config.num_hidden_layers)

arch_optimizer = torch.optim.Adam(
    arch_net.parameters(),
    lr=3e-4,
    betas=(0.5, 0.999)
)

outer = Arch(
    name="arch",
    module=arch_net,
    optimizer=arch_optimizer,
    train_data_loader=eval_dataloader,
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

engine = NASEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()




'''
# Get the metric function
if data_args.task_name is not None:
    metric = load_metric("glue", data_args.task_name)
# TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
# compute_metrics

# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    if data_args.task_name is not None:
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    elif is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

# Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
if data_args.pad_to_max_length:
    data_collator = default_data_collator
elif training_args.fp16:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
else:
    data_collator = None



# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Training
if training_args.do_train:
    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_name_or_path):
        # Check the config from that potential checkpoint has the right number of labels before using it as a
        # checkpoint.
        if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
            checkpoint = model_args.model_name_or_path

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# Evaluation
if training_args.do_eval:
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    eval_datasets = [eval_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(datasets["validation_mismatched"])

    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if training_args.do_predict:
    logger.info("*** Test ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    test_datasets = [test_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        test_datasets.append(datasets["test_mismatched"])

    for test_dataset, task in zip(test_datasets, tasks):
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        test_dataset.remove_columns_("label")
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

        output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logger.info(f"***** Test results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")
    '''


