#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prépare un dataset à partir de PleIAs/Post-OCR-Correction (choix de config: english, french, etc.)
- Télécharge une config (ex: english)
- Simule du bruit OCR : coupures de lignes + léger shuffle
- Fusionne avec tes fichiers existants (train.jsonl, val.jsonl)
- Sauvegarde train_merged.jsonl et val_merged.jsonl
"""

import argparse, json, random, re, sys
from pathlib import Path
from typing import Dict, Iterable, List
from datasets import load_dataset

INSTRUCTION = (
    "Rewrite this noisy OCR text into one clean, well-ordered paragraph. "
    "Fix casing, punctuation, and obvious spelling mistakes, and reorder lines if needed. "
    "Keep the meaning unchanged."
)

def split_into_pseudo_lines(s: str, linebreak_prob: float, seed: int) -> List[str]:
    rnd = random.Random(seed)
    words = s.split()
    if len(words) < 8:
        return [s]
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if rnd.random() < linebreak_prob and len(cur) >= 3:
            lines.append(" ".join(cur))
            cur = []
    if cur:
        lines.append(" ".join(cur))
    return lines

def slight_shuffle(lines: List[str], shuffle_prob: float, seed: int) -> List[str]:
    rnd = random.Random(seed + 1337)
    if len(lines) < 3 or shuffle_prob <= 0.0:
        return lines
    out, i = lines[:], 0
    while i < len(out) - 1:
        if rnd.random() < shuffle_prob:
            out[i], out[i+1] = out[i+1], out[i]
            i += 2
        else:
            i += 1
    return out

def make_example(noisy: str, clean: str, linebreak_prob: float, shuffle_prob: float, seed: int) -> Dict:
    noisy = re.sub(r"\s+", " ", noisy).strip()
    clean = re.sub(r"\s+", " ", clean).strip()
    if not noisy or not clean:
        return None
    lines = split_into_pseudo_lines(noisy, linebreak_prob, seed)
    lines = slight_shuffle(lines, shuffle_prob, seed)
    return {
        "instruction": INSTRUCTION,
        "input": "\n".join(lines),
        "output": clean
    }

def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def save_jsonl(path: Path, rows: Iterable[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config PleIAs (english, french, german, italian)")
    ap.add_argument("--train_in", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--val_in", type=Path, default=Path("data/val.jsonl"))
    ap.add_argument("--out_train", type=Path, default=Path("data/train_merged.jsonl"))
    ap.add_argument("--out_val", type=Path, default=Path("data/val_merged.jsonl"))
    ap.add_argument("--samples_train", type=int, default=2000)
    ap.add_argument("--samples_val", type=int, default=200)
    ap.add_argument("--linebreak_prob", type=float, default=0.35)
    ap.add_argument("--shuffle_prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"Loading PleIAs/Post-OCR-Correction config={args.config}…")
    ds = load_dataset("PleIAs/Post-OCR-Correction", args.config)

    # Détection des colonnes
    probe = None
    for sp in ("train","validation","test"):
        if sp in ds and len(ds[sp]) > 0:
            probe = ds[sp][0]
            break
    if probe is None:
        print("Dataset seems empty.", file=sys.stderr)
        sys.exit(2)

    candidate_pairs = [
        ("ocr", "corrected"),
        ("noisy", "clean"),
        ("ocr_text", "corrected_text"),
        ("text", "corrected_text"),      # <— clé rencontrée dans english
        ("text_ocr", "text_clean"),
        ("source", "target"),
        ("input", "output"),
        ("error_text", "ground_truth_text"),
    ]
    noisy_col = clean_col = None
    for a, b in candidate_pairs:
        if a in probe and b in probe:
            noisy_col, clean_col = a, b
            break
    if not noisy_col:
        print("Could not detect noisy/clean columns. Keys:", list(probe.keys()))
        sys.exit(3)

    # Collect PleIAs pairs
    def take_pairs(split_name: str, n: int):
        if split_name not in ds:
            return []
        out = []
        for ex in ds[split_name]:
            noisy = ex.get(noisy_col, "")
            clean = ex.get(clean_col, "")
            if isinstance(noisy, str) and isinstance(clean, str):
                out.append((noisy, clean))
                if len(out) >= n:
                    break
        return out

    train_pairs = take_pairs("train", args.samples_train)
    val_pairs = take_pairs("validation", args.samples_val)
    if not val_pairs:
        take = min(args.samples_val, max(1, len(train_pairs)//10))
        val_pairs = train_pairs[:take]
        train_pairs = train_pairs[take:]

    train_new = [make_example(n, c, args.linebreak_prob, args.shuffle_prob, args.seed+i) for i,(n,c) in enumerate(train_pairs)]
    val_new = [make_example(n, c, args.linebreak_prob, args.shuffle_prob, args.seed+10_000+i) for i,(n,c) in enumerate(val_pairs)]
    train_new = [x for x in train_new if x]
    val_new = [x for x in val_new if x]

    # Merge avec tes fichiers existants
    train_old = load_jsonl(args.train_in)
    val_old = load_jsonl(args.val_in)
    train_merged = train_old + train_new
    val_merged = val_old + val_new

    save_jsonl(args.out_train, train_merged)
    save_jsonl(args.out_val, val_merged)

    print(f"Saved merged train: {args.out_train} ({len(train_merged)} examples)")
    print(f"Saved merged val:   {args.out_val} ({len(val_merged)} examples)")

if __name__ == "__main__":
    main()
