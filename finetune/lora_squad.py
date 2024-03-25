import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    EvalPrediction,)
from utils_qa import postprocess_qa_predictions

from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from peft.utils import globals

model_name='roberta-base'
#model_name='microsoft/deberta-v3-base'

raw_datasets = load_dataset('squad')
dataset_train=raw_datasets['train']

#print(dataset_train[0])

tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cls_token="<s>",
    )

question_column_name='question'
context_column_name='context'
answer_column_name='answers'
pad_on_right = tokenizer.padding_side == "right"
max_seq_length=384 
doc_stride=128
batch_size=16
# Training preprocessing
def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" ,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

column_names = raw_datasets["validation"].column_names
print(column_names)
train_dataset = dataset_train.map(
                prepare_train_features,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )

device='cuda'
model = AutoModelForQuestionAnswering.from_pretrained(
    model_name
)

peft_config = LoraConfig(
    task_type=TaskType.QUESTION_ANS, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, bias="all"
)

model = get_peft_model(model, peft_config)

data_collator = default_data_collator
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size,drop_last=True)
#print(model(**next(iter(train_dataloader))))


# Validation preprocessing
def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

# We will select sample from whole data
dataset_eval = raw_datasets["validation"]
column_names = dataset_eval.column_names
eval_dataset = dataset_eval.map(
    prepare_validation_features,
    batched=True,
    remove_columns=column_names,
    desc="Running tokenizer on validation dataset",
)

lr=5e-4
optimizer = AdamW(params=model.parameters(), lr=lr)
model=model.to(device)


# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

eval_dataloader = DataLoader(eval_dataset.remove_columns(['offset_mapping','example_id']), shuffle=False, collate_fn=data_collator, batch_size=batch_size,drop_last=False)

metric = load_metric("squad")

def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids) 

model.train()
num_epochs=10
for e in range(num_epochs):
    #for i,batch in enumerate(tqdm(train_dataloader)):
    for i,batch in enumerate(train_dataloader):
        for k,v in batch.items():
            batch[k]=v.to(device)
        outputs = model(**batch)
        
        loss = outputs.loss
        #epoch_loss += batch_size * loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    predictions_start,predictions_end=[],[]

    num_eval=0
    model.eval()
    #for batch in tqdm(eval_dataloader):
    for batch in eval_dataloader:
        num_eval+=batch_size
        
        for k,v in batch.items():
            batch[k]=v.to(device)
        outputs = model(**batch)
        predictions_start.append(outputs.start_logits.cpu().detach().numpy())
        predictions_end.append(outputs.end_logits.cpu().detach().numpy())

    predictions=(np.concatenate(predictions_start),np.concatenate(predictions_end))
    examples=raw_datasets["validation"]
    features=eval_dataset
    eval_pred=post_processing_function(examples,features,predictions)
    
    print('Eval:',compute_metrics(eval_pred))


