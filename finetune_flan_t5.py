#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, inspect, torch, transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Trainer, Seq2SeqTrainingArguments,
)

# ====== ENV CONFIG ======
BASE_MODEL = os.getenv("BASE_MODEL", "google/flan-t5-base")  # base par défaut (plus léger)
TRAIN = os.getenv("TRAIN", "data/train_merged.jsonl")
VAL   = os.getenv("VAL",   "data/val_merged.jsonl")
OUT   = os.getenv("OUT",   "outputs/flan5-rewriter-base")
FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"               # set FORCE_CPU=1 pour forcer CPU

PROMPT = "{instruction}\n\nText:\n\n{input}\n\n"

def vtuple(v: str):
    parts = []
    for p in v.split("."):
        n = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(n) if n else 0)
    return tuple((parts + [0,0,0])[:3])

HF_VER = transformers.__version__

def log(x): print(f"[FT] {x}", flush=True)

def format_example(ex):
    return {
        "input_text": PROMPT.format(instruction=ex["instruction"], input=ex["input"]).strip(),
        "target_text": ex["output"].strip()
    }

def main():
    log(f"transformers={HF_VER} | torch={torch.__version__}")
    log(f"BASE_MODEL={BASE_MODEL}")
    log(f"TRAIN={TRAIN}")
    log(f"VAL={VAL}")
    log(f"OUT={OUT}")

    # ===== Device =====
    use_cuda = torch.cuda.is_available()
    use_mps  = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    if FORCE_CPU:
        device = "cpu"
    else:
        device = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
    log(f"device={device}")
    if device == "mps":
        # Optionnel : lever la limite mémoire MPS (à tes risques)
        if os.getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO") is None:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # ===== Modèle & tokenizer =====
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    # Réduire l'empreinte mémoire en entraînement
    model.config.use_cache = False                 # IMPORTANT pour checkpointing
    model.gradient_checkpointing_enable()          # réduit fortement la RAM/VGRAM

    # ===== Dataset =====
    ds = load_dataset("json", data_files={"train": TRAIN, "validation": VAL}).map(format_example)

    # Tailles plus petites pour MPS/peu de RAM
    def tokenize(batch):
        x = tok(batch["input_text"], truncation=True, max_length=512)   # avant 1024
        y = tok(text_target=batch["target_text"], truncation=True, max_length=128)  # avant 512
        x["labels"] = y["input_ids"]
        return x

    ds_tok = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorForSeq2Seq(tok, model=model)

    # ===== Args low-RAM =====
    kwargs = dict(
        output_dir=OUT,
        learning_rate=2e-4,
        per_device_train_batch_size=1,       # batch min
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,      # simule batch global 32
        num_train_epochs=2,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False,         # MPS: inutile, évite warning
        # Pas d'éval en ligne (économise VRAM)
        # predict_with_generate True peut déclencher de la génération -> on évite pendant train
    )

    # Compat éventuelle si tu veux remettre une éval plus tard :
    # sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    # if "eval_strategy" in sig.parameters:
    #     kwargs["eval_strategy"] = "no"
    # elif "evaluation_strategy" in sig.parameters:
    #     kwargs["evaluation_strategy"] = "no"

    # bf16/fp16: pas sur MPS. Sur CUDA, on peut activer.
    if device == "cuda":
        try:
            major = torch.cuda.get_device_capability(0)[0]
            if major >= 8:
                kwargs["bf16"] = True
            else:
                kwargs["fp16"] = True
        except Exception:
            pass

    # Filtre de sécurité : retirer toute clé inconnue
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    args = Seq2SeqTrainingArguments(**kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=None,                  # pas d'éval pendant train
        tokenizer=tok,                      # futurewarning ok
        data_collator=collator,
    )

    # ===== Entraînement =====
    trainer.train()

    # Éval post-train légère (optionnelle)
    # try:
    #     from transformers import EvalPrediction
    #     trainer.evaluate()   # sans generation pour rester léger
    # except Exception as e:
    #     log(f"Evaluation skipped: {e}")

    # ===== Sauvegarde =====
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    log(f"✅ checkpoint: {OUT}")

if __name__ == "__main__":
    main()
