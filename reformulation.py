#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, gc, unicodedata
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

STOP_TRIGGERS = re.compile(
    r"(?is)\b(rule|rules|instruction|guideline|rewrite|rephrase|edit|output:?|response:?|answer:?|text:)\b"
)

def load_reconstructed_text(inp: Path) -> str:
    if inp.is_dir():
        p = inp / "text_final.txt"
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
        pj = inp / "reconstructed.json"
        if pj.exists():
            data = json.loads(pj.read_text(encoding="utf-8"))
            return str(data.get("final_text", "")).strip()
        raise SystemExit(f"[error] Neither text_final.txt nor reconstructed.json found in: {inp}")
    if inp.suffix.lower() == ".txt":
        return inp.read_text(encoding="utf-8").strip()
    if inp.suffix.lower() == ".json":
        data = json.loads(inp.read_text(encoding="utf-8"))
        return str(data.get("final_text", "")).strip()
    raise SystemExit("[error] --input must be a folder or a .txt/.json file produced by reconstruct_sentences.py")

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def basic_cleanup(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,;:.!?])", r"\1", t)
    t = re.sub(r"([.?!]){2,}", r".", t)
    t = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "o", t)
    t = re.sub(r"(?<=[A-Za-z])1(?=[A-Za-z])", "l", t)
    return t.strip()

def sentence_split(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def sanitize_line(s: str) -> str:
    s = s.strip()
    # Couper dès qu'on voit un mot qui rappelle le prompt
    m = STOP_TRIGGERS.search(s)
    if m:
        s = s[:m.start()].strip()
    # Retirer éventuelles étiquettes
    s = re.sub(r"(?i)^\s*(rewritten|output|response|answer|edited)\s*:?\s*", "", s)
    # Retirer guillemets parasites
    s = s.strip(" '\"“”‘’`")
    # ASCII basique
    s = re.sub(r"[^\x20-\x7E]", "", s)
    # Espace propre
    s = re.sub(r"\s+", " ", s)
    # Première lettre en majuscule si phrase
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    # Finir par point si c'est une phrase
    if s and not re.search(r"[.!?]$", s):
        s += "."
    return s

def compress_for_bullet(s: str, n: int = 8) -> str:
    s = s.strip().rstrip(".!?")
    toks = s.split()
    return " ".join(toks[:n])

def pick_title(text: str) -> str:
    s = text.lower()
    m = re.search(r"\b(paris|london|rome|madrid|berlin|tokyo|new york)\b", s)
    if m:
        return f"Daily Notes — {m.group(1).title()}"
    return "Daily Life"

def load_model(force_cpu: bool):
    if not force_cpu and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        device = "mps"
    else:
        device = "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, low_cpu_mem_usage=True).to(device)
    return tok, mdl, device

def build_chat_prompt(tok, sentence: str) -> str:
    user = (
        "Fix the sentence in simple English. Keep the meaning. "
        "Do not add details. Write only the corrected sentence. English only.\n"
        f"Sentence: {sentence}"
    )
    if hasattr(tok, "apply_chat_template"):
        msgs = [{"role":"user","content":user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Fallback texte brut minimal
    return user + "\nAnswer:"

def rewrite_sentence_en(tok, mdl, device: str, sent: str, max_new_tokens: int = 30) -> str:
    prompt = build_chat_prompt(tok, sent)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out_ids = mdl.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy
            temperature=0.0,
            top_p=1.0,
            no_repeat_ngram_size=4,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    # enlever le prompt si causal
    gen_only = out_ids[0][enc["input_ids"].shape[-1]:]
    txt = tok.decode(gen_only, skip_special_tokens=True)
    return sanitize_line(txt)

def build_markdown(title: str, sentences: list[str]) -> str:
    # Paragraph: 2–5 phrases max
    lesson_sents = [s for s in sentences if s][:5]
    para = " ".join(lesson_sents[:5]).strip()
    if para and not para.endswith("."):
        para += "."
    # Bullets: premières phrases
    bullets, seen = [], set()
    for s in lesson_sents:
        b = compress_for_bullet(s, 8)
        b_low = b.lower()
        if b and b_low not in seen:
            bullets.append(b)
            seen.add(b_low)
        if len(bullets) >= 5:
            break
    if not bullets and sentences:
        bullets = [compress_for_bullet(sentences[0], 8)]
    md = [
        f"# {title}",
        "",
        "## Lesson",
        para if para else "No clear content.",
        "",
        "## Key Points",
    ]
    md += [f"- {b}" for b in bullets]
    return "\n".join(md).strip() + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_sentences", type=int, default=6)
    ap.add_argument("--force_cpu", action="store_true")
    args = ap.parse_args()

    raw = load_reconstructed_text(Path(args.input))
    if not raw.strip():
        raise SystemExit("[error] empty reconstructed text")

    pre = strip_accents(raw)
    pre = basic_cleanup(pre)
    sents = sentence_split(pre)[: max(1, args.max_sentences)]

    tok = mdl = None
    out_path = Path(args.out)
    try:
        tok, mdl, dev = load_model(args.force_cpu)
        rewritten = []
        for s in sents:
            if not s or len(s) > 300:
                continue
            y = rewrite_sentence_en(tok, mdl, dev, s)
            # si la phrase contient encore un trigger, on coupe tout après le point précédent
            if STOP_TRIGGERS.search(y):
                y = y.split(".")[0].strip() + "."
            if y:
                rewritten.append(y)

        if not rewritten:
            rewritten = sents  # fallback très sobre

        title = pick_title(" ".join(rewritten))
        md = build_markdown(title, rewritten)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"[ok] {out_path}")
    finally:
        try: del tok, mdl
        except Exception: pass
        gc.collect()
        if torch.backends.mps.is_available():
            try: torch.mps.empty_cache()
            except Exception: pass

if __name__ == "__main__":
    main()
