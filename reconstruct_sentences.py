#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reconstruct_sentences.py (LLM post-edit, aligné au fine-tune + fallback amélioré)
- Lit un texte OCR depuis ocr_results.json
- Envoie le texte à un modèle seq2seq (prompt cohérent avec le fine-tune)
- Écrit le texte final (et un titre optionnel)
- Préserve toutes les lignes; fallback ligne-à-ligne si besoin (consigne plus "prof")
- Fallback gracieux si transformers est manquant (retourne l'OCR pré-nettoyé)

Usage:
  python3 reconstruct_sentences.py <folder|manifest.json> [--ocr OCR_PATH]
    [--fix_model ouiyam/see4me-flan5-rewriter-base] [--max_new_tokens 160]
    [--title] [--title_model MODEL] [--title_max_new_tokens 16]
    [--no_tiny_cleanup] [--force_per_line]
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

_HAS_TRANSFORMERS = False
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


# ====== Prompts ======
# Instruction globale (même esprit que le dataset de fine-tune) + wrapper Text:
INSTRUCTION_EN = (
    "Rewrite this noisy OCR text into one clean, well-ordered paragraph. "
    "Fix casing, punctuation, and obvious spelling mistakes, and reorder lines if needed. "
    "Keep the meaning unchanged. Output in English."
)

# Pour le titre (optionnel)
PROMPT_TITLE_EN = (
    "Suggest a concise (≤8 words), informative, natural title for this text. "
    "Answer with the title only:\n\n"
)

def make_fix_prompt(text: str) -> str:
    """Construit le prompt EXACTEMENT comme au fine-tune : instruction + 'Text:' + texte."""
    return f"{INSTRUCTION_EN}\n\nText:\n\n{text}\n\n"


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
    """Nettoyage déterministe après LLM pour ôter quelques bruits OCR courants."""
    if not s:
        return s
    s = re.sub(r"^\s*Output:\s*", "", s, flags=re.IGNORECASE)  # remove 'Output:' prefix if echoed
    s = re.sub(r"\b0 0\b", "", s)                              # remove '0 0' noise
    s = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "o", s)
    s = re.sub(r"(?<=[A-Za-z])1(?=[A-Za-z])", "l", s)
    # remove isolated single letters that stand alone as tokens (common stray 't', 'i', etc.)
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
        # Fallback: retourner le bloc après "Text:\n" si transformers indisponible
        if "\n\nText:\n\n" in prompt:
            return prompt.split("\n\nText:\n\n", 1)[-1].strip()
        return prompt.strip()
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
        # Si quelque chose échoue, renvoyer le texte brut (mieux que planter)
        if "\n\nText:\n\n" in prompt:
            return prompt.split("\n\nText:\n\n", 1)[-1].strip()
        return prompt.strip()


def fix_per_line(text: str, model_name: str, max_new_tokens: int) -> str:
    """
    Édite chaque ligne indépendamment en conservant le format de prompt du fine-tune.
    Version "prof" : autorise la complétion minimale (sujet/auxiliaire manquant) et corrige OCR évident.
    """
    lines = text.splitlines()
    out_lines = []
    for ln in lines:
        if not ln.strip():
            out_lines.append("")  # préserver les lignes vides
            continue
        prompt = (
            "Rewrite this single OCR line into correct English as a standalone sentence. "
            "Fix casing, grammar, punctuation, and obvious OCR mistakes (e.g., 0→o, 1→l, 'citey'→'city'). "
            "If the subject or auxiliary is clearly missing, add the minimal natural completion. "
            "Return ONLY the corrected sentence.\n\n"
            "Text:\n\n"
            f"{ln.strip()}\n\n"
        )
        y = run_seq2seq(prompt, model_name, max_new_tokens)
        y = re.sub(r"^\s*Output:\s*", "", y, flags=re.IGNORECASE).strip()
        out_lines.append(y)
    return "\n".join(out_lines)


# ====== Main ======
def main():
    ap = argparse.ArgumentParser(description="Minimal LLM post-edit on OCR text (line-preserving), prompt aligné au fine-tune.")
    ap.add_argument("input", help="Dossier avec manifest.json + ocr_results.json OU chemin direct vers manifest.json")
    ap.add_argument("--ocr", help="Chemin vers ocr_results.json (sinon inféré)")

    # Modèle & génération
    ap.add_argument("--fix_model", default="ouiyam/see4me-flan5-rewriter-base", help="Checkpoint seq2seq fine-tuné pour la réécriture")
    ap.add_argument("--max_new_tokens", type=int, default=160, help="Max new tokens pour la génération (160 conseillé)")

    # Titre (optionnel)
    ap.add_argument("--title", action="store_true", help="Générer aussi un titre")
    ap.add_argument("--title_model", default=None, help="Modèle pour le titre (par défaut: fix_model)")
    ap.add_argument("--title_max_new_tokens", type=int, default=16, help="Max new tokens pour le titre")

    # Post-processing
    ap.add_argument("--no_tiny_cleanup", action="store_true", help="Désactiver le tiny cleanup après LLM")

    # Mode ligne-à-ligne explicite
    ap.add_argument("--force_per_line", action="store_true", help="Forcer l'édition par ligne (skip le passage global)")

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

    # Métadonnées pour la sortie
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1) Lire le texte OCR
    raw_text = load_ocr_text(ocr_path)
    pre = light_preclean(raw_text)

    # 2) LLM copyedit
    if args.force_per_line:
        print(f"[fix] per-line mode | model={args.fix_model} | max_new_tokens={args.max_new_tokens}")
        final_text = fix_per_line(pre, args.fix_model, args.max_new_tokens)
    else:
        fix_prompt = make_fix_prompt(pre)  # <<< Aligné avec le fine-tune
        print(f"[fix] model={args.fix_model} | max_new_tokens={args.max_new_tokens}")
        final_text = run_seq2seq(fix_prompt, args.fix_model, args.max_new_tokens)
        # Retirer un éventuel 'Output:' préfixé par le modèle
        final_text = re.sub(r"^\s*Output:\s*", "", final_text, flags=re.IGNORECASE).strip()

        # 2b) Fallback ligne-à-ligne UNIQUEMENT si sortie vraiment trop courte
        # (on n'utilise plus l'écart de lignes pour forcer le fallback)
        too_short = len(final_text) < 0.3 * len(pre)
        if too_short:
            print("[fix] Fallback: per-line mode (very short output)")
            final_text = fix_per_line(pre, args.fix_model, args.max_new_tokens)

    # 3) Nettoyage déterministe optionnel
    if not args.no_tiny_cleanup:
        final_text = tiny_cleanup(final_text)

    # 4) Titre optionnel
    title = ""
    if args.title:
        model_t = args.title_model or args.fix_model
        print(f"[title] model={model_t} | max_new_tokens={args.title_max_new_tokens}")
        # Ici, le prompt titre n’a pas été fine-tuné — on l’assume comme tâche distincte
        title = run_seq2seq(PROMPT_TITLE_EN + final_text, model_t, args.title_max_new_tokens)
        title = re.sub(r"^\s*Output:\s*", "", title, flags=re.IGNORECASE).strip()

    # 5) Sauvegarde
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
