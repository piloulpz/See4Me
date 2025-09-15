#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reconstruct_sentences.py (minimal LLM post-edit, robust)
- Reads OCR text from ocr_results.json
- Sends it to a seq2seq LLM with a copyediting prompt
- Writes final text (and optional title)
- Preserves ALL lines; falls back to per-line editing if needed
- Graceful fallback if transformers is missing (returns precleaned OCR)

Usage:
  python3 reconstruct_sentences.py <folder|manifest.json> [--ocr OCR_PATH]
    [--fix_model google/flan-t5-large] [--max_new_tokens 512]
    [--title] [--title_model MODEL] [--title_max_new_tokens 32]
    [--no_tiny_cleanup]
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Optional

_HAS_TRANSFORMERS = False
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


# ====== Prompts ======
# Global prompt: insist on preserving ALL lines and counts.
PROMPT_FIX_EN = (
    "Fix the following text in English. You have to detect all the words and anomalies in the sentences. Put them together to make the text readble\n"
    "Delete all the sentences which are not english\n"
    "Text :\n\n"
)

PROMPT_TITLE_EN = (
    "Suggest a concise (≤8 words), informative, natural title for this text. "
    "Answer with the title only:\n\n"
)


# ====== I/O helpers ======
def load_ocr_text(ocr_path: Path) -> str:
    data = json.loads(ocr_path.read_text(encoding="utf-8"))
    if isinstance(data.get("full_text"), str) and data["full_text"].strip():
        return data["full_text"].strip()
    results = data.get("results", [])
    lines = [r.get("text", "") for r in results if isinstance(r.get("text", ""), str)]
    return "\n".join([t for t in lines if t.strip()]).strip()


# ====== Light cleanup (pre & post) ======
def light_preclean(t: str) -> str:
    if not t:
        return t
    s = t.replace("\u00a0", " ")  # NBSP -> space
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+([,;:.!?])", r"\1", s)   # no space before punctuation
    s = re.sub(r"([.?!]){2,}", r".", s)      # collapse repeated end-punct
    # common OCR digit/letter confusions inside words
    s = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "o", s)
    s = re.sub(r"(?<=[A-Za-z])1(?=[A-Za-z])", "l", s)
    # normalize weird dashes to hyphen
    s = s.replace("–", "-").replace("—", "-").replace("-", "-")
    return s.strip()


def tiny_cleanup(s: str) -> str:
    """Small deterministic cleanup after the LLM to guarantee removal of common OCR noise."""
    if not s:
        return s
    s = re.sub(r"^\s*Output:\s*", "", s, flags=re.IGNORECASE)  # remove 'Output:' prefix if echoed
    s = re.sub(r"\b0 0\b", "", s)                              # remove '0 0' noise
    s = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "o", s)
    s = re.sub(r"(?<=[A-Za-z])1(?=[A-Za-z])", "l", s)
    # remove isolated single letters that stand alone as tokens (common stray 't', 'i' etc.)
    s = re.sub(r"(?:^|\s)\b([A-Za-z])\b(?:[. ]+|$)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Capitalize sentence starts
    def cap(m):
        return m.group(1) + m.group(2).upper()
    s = re.sub(r"(^|[.!?]\s+)([a-z])", cap, s)
    return s


# ====== LLM runner ======
def run_seq2seq(prompt: str, model_name: str, max_new_tokens: int) -> str:
    if not _HAS_TRANSFORMERS:
        # Fallback: return the input text part (after "Text:\n") if transformers is unavailable
        return prompt.split("\n\nText:\n", 1)[-1].strip()
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        inp = tok(prompt, return_tensors="pt", truncation=True)
        ids = mdl.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            length_penalty=0.9,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return tok.decode(ids[0], skip_special_tokens=True).strip()
    except Exception:
        # If anything fails, return the raw text (still better than crashing)
        return prompt.split("\n\nText:\n", 1)[-1].strip()


def fix_per_line(text: str, model_name: str, max_new_tokens: int) -> str:
    """Edit each line independently to strictly preserve line count."""
    lines = text.splitlines()
    out_lines = []
    for ln in lines:
        if not ln.strip():
            out_lines.append("")  # preserve blank lines
            continue
        prompt = (
            "Edit ONLY this one line. Keep its meaning and keep all words unless they are clear OCR noise. "
            "Fix spelling, grammar, casing, spacing, and punctuation; remove OCR noise (0→o, 1→l). "
            "Return ONLY the corrected line:\n\n"
            f"{ln.strip()}"
        )
        y = run_seq2seq(prompt, model_name, max_new_tokens)
        y = re.sub(r"^\s*Output:\s*", "", y, flags=re.IGNORECASE).strip()
        out_lines.append(y)
    return "\n".join(out_lines)


# ====== Main ======
def main():
    ap = argparse.ArgumentParser(description="Minimal LLM post-edit on OCR text (line-preserving).")
    ap.add_argument("input", help="Folder with manifest.json + ocr_results.json OR the manifest.json path")
    ap.add_argument("--ocr", help="Path to ocr_results.json (else inferred)")

    # Models & gen params
    ap.add_argument("--fix_model", default="google/flan-t5-large", help="Seq2seq model for copyediting")
    ap.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for the copyedit model")

    # Optional title
    ap.add_argument("--title", action="store_true", help="Also generate a title")
    ap.add_argument("--title_model", default=None, help="Optional different model for title generation")
    ap.add_argument("--title_max_new_tokens", type=int, default=32, help="Max new tokens for title")

    # Post-processing
    ap.add_argument("--no_tiny_cleanup", action="store_true", help="Disable tiny cleanup after LLM")

    # Optional explicit per-line mode
    ap.add_argument("--force_per_line", action="store_true", help="Force per-line editing (skip global pass)")

    args = ap.parse_args()

    p = Path(args.input)
    if p.is_dir():
        manifest_path = p / "manifest.json"
        ocr_path = Path(args.ocr) if args.ocr else (p / "ocr_results.json")
    elif p.is_file() and p.name == "manifest.json":
        manifest_path = p
        ocr_path = Path(args.ocr) if args.ocr else (p.parent / "ocr_results.json")
    else:
        raise SystemExit("Input must be a folder or a manifest.json file.")

    if not manifest_path.exists():
        raise SystemExit(f"manifest.json not found: {manifest_path}")
    if not ocr_path.exists():
        raise SystemExit(f"ocr_results.json not found: {ocr_path}")

    # Load for output metadata
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1) Read OCR text
    raw_text = load_ocr_text(ocr_path)
    pre = light_preclean(raw_text)

    # 2) LLM copyedit
    if args.force_per_line:
        print(f"[fix] per-line mode | model={args.fix_model} | max_new_tokens={args.max_new_tokens}")
        final_text = fix_per_line(pre, args.fix_model, args.max_new_tokens)
    else:
        fix_prompt = PROMPT_FIX_EN + pre
        print(f"[fix] model={args.fix_model} | max_new_tokens={args.max_new_tokens}")
        final_text = run_seq2seq(fix_prompt, args.fix_model, args.max_new_tokens)
        # Remove possible 'Output:' prefix
        final_text = re.sub(r"^\s*Output:\s*", "", final_text, flags=re.IGNORECASE).strip()
        # 2b) Fallback to per-line if the model has summarized or dropped lines
        input_lines = [l for l in pre.splitlines()]
        output_lines = [l for l in final_text.splitlines()]
        too_short = len(final_text) < 0.6 * len(pre)
        lost_many_lines = abs(len(output_lines) - len(input_lines)) > max(2, int(0.1 * len(input_lines)))
        if too_short or lost_many_lines:
            print("[fix] Fallback: per-line mode (output too short or line count mismatch)")
            final_text = fix_per_line(pre, args.fix_model, args.max_new_tokens)

    # 3) Optional tiny cleanup (deterministic)
    if not args.no_tiny_cleanup:
        final_text = tiny_cleanup(final_text)

    # 4) Optional title
    title = ""
    if args.title:
        model_t = args.title_model or args.fix_model
        print(f"[title] model={model_t} | max_new_tokens={args.title_max_new_tokens}")
        title = run_seq2seq(PROMPT_TITLE_EN + final_text, model_t, args.title_max_new_tokens)

    # 5) Save
    out_dir = manifest_path.parent
    out = {
        "source_image": manifest.get("source_image"),
        "full_text_ocr": raw_text,
        "final_text": final_text,
        "title": title,
        "fix_model": args.fix_model,
    }
    (out_dir / "reconstructed.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "text_final.txt").write_text(final_text + "\n", encoding="utf-8")
    if title:
        (out_dir / "title.txt").write_text(title + "\n", encoding="utf-8")

    print("\n=== TITLE ===")
    print(title or "(none)")
    print("\n=== FINAL TEXT ===")
    print(final_text)
    print(f"\n[reconstruct] Output: {out_dir/'reconstructed.json'} | text_final.txt | title.txt")


if __name__ == "__main__":
    main()
