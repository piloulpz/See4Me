#!/usr/bin/env python3
import os, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments

BASE_MODEL = os.getenv("BASE_MODEL", "google/flan-t5-large")
TRAIN = os.getenv("TRAIN", "data/train_merged.jsonl")
VAL   = os.getenv("VAL",   "data/val_merged.jsonl")
OUT   = os.getenv("OUT",   "outputs/flan5-rewriter")

# même esprit que reconstruct: Instruction + Input -> Output
PROMPT = "{instruction}\n\nText:\n\n{input}\n\n"

def format_example(ex):
    ins, inp, out = ex["instruction"], ex["input"], ex["output"]
    src = PROMPT.format(instruction=ins, input=inp).strip()
    tgt = out.strip()
    return {"input_text": src, "target_text": tgt}

ds = load_dataset("json", data_files={"train": TRAIN, "validation": VAL})
ds = ds.map(format_example)

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

def tokenize(b):
    x = tok(b["input_text"], truncation=True, max_length=1024)
    y = tok(text_target=b["target_text"], truncation=True, max_length=512)
    x["labels"] = y["input_ids"]
    return x

ds_tok = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
collator = DataCollatorForSeq2Seq(tok, model=model)

args = TrainingArguments(
    output_dir=OUT,
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    bf16=True,
    logging_steps=25,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    predict_with_generate=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    tokenizer=tok,
    data_collator=collator,
)

trainer.train()
trainer.save_model(OUT)
tok.save_pretrained(OUT)
print("✅ checkpoint:", OUT)
