#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np

# PONCTUATION (optionnelle)
_HAS_TRANSFORMERS = False
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    pass


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self): return max(0, self.x1 - self.x0)
    @property
    def h(self): return max(0, self.y1 - self.y0)
    @property
    def cx(self): return self.x0 + self.w / 2.0
    @property
    def cy(self): return self.y0 + self.h / 2.0


@dataclass
class Token:
    text: str
    box: Box
    path: str


def load_inputs(manifest_path: Path, ocr_path: Path) -> List[Token]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    # Associer par chemin (clé)
    text_by_path = { Path(r["path"]).as_posix(): r["text"] for r in ocr.get("results", []) }

    tokens: List[Token] = []
    for c in manifest.get("crops", []):
        p = Path(c["path"]).as_posix()
        t = text_by_path.get(p, "")
        if not isinstance(t, str):
            t = ""
        # Décomposer en "mots" (simple split). On garde quand même le crop entier comme *ligne* si nécessaire.
        # Ici : on considère le crop comme un token (mot/mini-ligne) pour rester générique,
        # car tu as déjà segmenté en "word" ou "line" côté segmenter.
        if t.strip() == "":
            continue
        x0, y0, x1, y1 = c["bbox_abs"]
        tokens.append(Token(text=t.strip(), box=Box(int(x0), int(y0), int(x1), int(y1)), path=p))
    return tokens, manifest


def auto_columns(tokens: List[Token], n_columns: Optional[int] = None) -> List[int]:
    """
    Retourne un label de colonne pour chaque token.
    Heuristique 1D : on trie par x0 et on découpe selon les grands "sauts".
    Si n_columns est fixé, on coupe en quantiles; sinon on détecte automatiquement via un seuil sur les gaps.
    """
    if not tokens:
        return []

    order = np.argsort([t.box.x0 for t in tokens])
    x0_sorted = np.array([tokens[i].box.x0 for i in order], dtype=float)
    gaps = np.diff(x0_sorted)

    if n_columns and n_columns > 1:
        # couper en n_columns via quantiles des x0
        q = np.linspace(0, 1, n_columns + 1)
        bounds = np.quantile(x0_sorted, q)
        cols = np.zeros(len(tokens), dtype=int)
        for rank, i in enumerate(order):
            x = tokens[i].box.x0
            # trouver l'intervalle
            k = np.searchsorted(bounds, x, side="right") - 1
            k = min(max(k, 0), n_columns - 1)
            cols[i] = k
        return cols.tolist()

    # auto: grand gap = nouvelle colonne
    if len(gaps) == 0:
        return [0] * len(tokens)

    med = float(np.median(gaps))
    iqr = float(np.percentile(gaps, 75) - np.percentile(gaps, 25))
    thresh = med + 2.5 * iqr if iqr > 0 else med * 2.0
    thresh = max(thresh, 40.0)  # garde-fou absolu

    labels = np.zeros(len(tokens), dtype=int)
    col = 0
    prev = x0_sorted[0]
    for pos, i in enumerate(order):
        if pos > 0 and (x0_sorted[pos] - prev) > thresh:
            col += 1
        labels[i] = col
        prev = x0_sorted[pos]
    return labels.tolist()


def group_lines(tokens: List[Token], line_tol: float = 0.6) -> List[List[int]]:
    """
    Regroupe en lignes en se basant sur la proximité verticale des centres Y + chevauchement vertical.
    line_tol est un facteur *hauteur moyenne* pour tolérer les variations.
    Retourne une liste de groupes, chaque groupe est une liste d'indices dans 'tokens'.
    """
    if not tokens:
        return []

    # Trier par centre Y
    idxs = list(range(len(tokens)))
    idxs.sort(key=lambda i: tokens[i].box.cy)

    # hauteur médiane pour seuils
    heights = np.array([tokens[i].box.h for i in idxs], dtype=float)
    h_med = float(np.median(heights)) if heights.size else 20.0
    band = max(10.0, line_tol * h_med)

    groups: List[List[int]] = []
    current: List[int] = []
    last_cy = None

    for i in idxs:
        cy = tokens[i].box.cy
        if last_cy is None:
            current = [i]
            last_cy = cy
            continue
        # même ligne si |Δcy| <= band OU si fort chevauchement vertical
        same_line = abs(cy - last_cy) <= band
        if not same_line:
            # nouveau groupe
            groups.append(sorted(current, key=lambda j: tokens[j].box.x0))
            current = [i]
        else:
            current.append(i)
        last_cy = cy

    if current:
        groups.append(sorted(current, key=lambda j: tokens[j].box.x0))

    return groups


def lines_to_paragraphs(lines: List[List[int]], tokens: List[Token], para_factor: float = 1.6) -> List[List[List[int]]]:
    """
    Regroupe des lignes en paragraphes selon le gap vertical inter-lignes.
    para_factor * median_line_gap sert de seuil.
    """
    if not lines:
        return []

    # Gap vertical entre lignes = distance entre y0 de la ligne N+1 et y1 de la ligne N
    def line_top(g): return min(tokens[i].box.y0 for i in g)
    def line_bottom(g): return max(tokens[i].box.y1 for i in g)

    line_tops = [line_top(g) for g in lines]
    line_bottoms = [line_bottom(g) for g in lines]
    gaps = [max(0, line_tops[i+1] - line_bottoms[i]) for i in range(len(lines)-1)]
    med_gap = float(np.median(gaps)) if gaps else 0.0
    thresh = (para_factor * med_gap) if med_gap > 0 else 30.0  # seuil de base

    paragraphs: List[List[List[int]]] = []
    current: List[List[int]] = [lines[0]]
    for i in range(1, len(lines)):
        gap = max(0, line_tops[i] - line_bottoms[i-1])
        if gap > thresh:
            paragraphs.append(current)
            current = [lines[i]]
        else:
            current.append(lines[i])
    if current:
        paragraphs.append(current)
    return paragraphs


def reconstruct_line(words: List[Token], space_factor: float = 0.33) -> str:
    """
    Reconstruit une ligne : insertion d'espaces selon gap relatif aux largeurs des mots.
    - On insère un espace si gap_x > space_factor * median_word_width.
    - Gestion simple des césures: si mot se termine par '-' et que le suivant commence en minuscule, on colle.
    """
    if not words:
        return ""

    # Ordre gauche->droite
    words = sorted(words, key=lambda t: t.box.x0)
    widths = np.array([w.box.w for w in words], dtype=float)
    med_w = float(np.median(widths)) if widths.size else 20.0
    pieces = [words[0].text]

    for prev, cur in zip(words[:-1], words[1:]):
        gap = cur.box.x0 - prev.box.x1
        need_space = gap > (space_factor * med_w)

        if pieces[-1].endswith("-") and cur.text and cur.text[0].islower():
            # césure : pas d'espace et retire '-'
            pieces[-1] = pieces[-1][:-1] + cur.text
        else:
            if need_space:
                pieces.append(" " + cur.text)
            else:
                # souvent collé = fin de mot + début de mot → insérer espace min si
                # le dernier char est lettre/chiffre et le premier aussi
                if pieces[-1] and pieces[-1][-1].isalnum() and cur.text[0].isalnum():
                    pieces.append(" " + cur.text)
                else:
                    pieces.append(cur.text)

    return "".join(pieces).strip()


def apply_punctuation(text: str,
                      model_name: str = "oliverguhr/fullstop-punctuation-multilang-large",
                      device: int = -1) -> str:
    """
    Applique une restauration de ponctuation simple via un modèle token-classification.
    Ce modèle prédit des tags type 'O', 'COMMA', 'PERIOD', 'QUESTION', ... après chaque jeton.
    Si indisponible, renvoie le texte inchangé.
    """
    if not _HAS_TRANSFORMERS:
        return text
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForTokenClassification.from_pretrained(model_name)
        nlp = pipeline("token-classification", model=mdl, tokenizer=tok, aggregation_strategy="simple", device=device)
    except Exception:
        return text

    # Tokeniser par espaces (le modèle s’en sort généralement)
    words = text.split()
    if not words:
        return text

    # Limiter la longueur par batch si nécessaire
    out_tokens = []
    batch_size = 128
    for s in range(0, len(words), batch_size):
        seg = words[s:s+batch_size]
        res = nlp(" ".join(seg))
        # res = liste de dicts avec 'word' et 'entity_group'
        punct_map = {
            "COMMA": ",",
            "PERIOD": ".",
            "QUESTION": "?",
            "EXCLAMATION": "!",
            "COLON": ":",
            "SEMICOLON": ";"
        }
        # On reconstruit : mot + ponctuation si prédite
        for r in res:
            w = r.get("word", "")
            ent = r.get("entity_group", "O")
            p = punct_map.get(ent, "")
            if w:
                out_tokens.append(w + p)
    # Post: espace après .,? etc. (sauf si déjà présent)
    txt = " ".join(out_tokens)
    txt = txt.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?").replace(" :", ":").replace(" ;", ";")
    return txt


def main():
    ap = argparse.ArgumentParser(description="Reconstruction de phrases à partir des crops + OCR (mot/ligne).")
    ap.add_argument("input", help="Dossier des crops (contenant manifest.json et ocr_results.json) OU chemin vers manifest.json")
    ap.add_argument("--ocr", help="Chemin vers ocr_results.json (sinon déduit du dossier/manifest)")
    ap.add_argument("--n_columns", type=int, default=None, help="Nombre de colonnes (auto si non fourni)")
    ap.add_argument("--line_tol", type=float, default=0.6, help="Tolérance de regroupement vertical (× hauteur médiane)")
    ap.add_argument("--space_factor", type=float, default=0.33, help="Seuil espace = factor × largeur médiane du mot")
    ap.add_argument("--para_factor", type=float, default=1.6, help="Seuil paragraphe = factor × gap médian inter-lignes")
    ap.add_argument("--punct", action="store_true", help="Activer restauration de ponctuation")
    ap.add_argument("--punct_model", default="oliverguhr/fullstop-punctuation-multilang-large", help="Nom du modèle de ponctuation HF")
    ap.add_argument("--device", type=int, default=-1, help="Device HF (-1 CPU, >=0 GPU/MPS si supporté)")
    args = ap.parse_args()

    p = Path(args.input)
    if p.is_dir():
        manifest_path = p / "manifest.json"
        ocr_path = Path(args.ocr) if args.ocr else (p / "ocr_results.json")
    elif p.is_file() and p.name == "manifest.json":
        manifest_path = p
        ocr_path = Path(args.ocr) if args.ocr else (p.parent / "ocr_results.json")
    else:
        raise SystemExit("Input doit être un dossier de crops ou un manifest.json")

    if not manifest_path.exists():
        raise SystemExit(f"manifest.json introuvable: {manifest_path}")
    if not ocr_path.exists():
        raise SystemExit(f"ocr_results.json introuvable: {ocr_path}")

    tokens, manifest = load_inputs(manifest_path, ocr_path)
    if not tokens:
        print("[reconstruct] Aucun token OCR exploitable.")
        out = {
            "source_image": manifest.get("source_image"),
            "reconstructed": [],
            "full_text_no_punct": "",
            "full_text": ""
        }
        out_path = manifest_path.parent / "reconstructed.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[reconstruct] Sortie: {out_path}")
        return

    # COLONNES
    cols = auto_columns(tokens, n_columns=args.n_columns)
    n_cols = max(cols) + 1 if cols else 1

    # Groupes par colonne
    col_groups: List[List[int]] = [[] for _ in range(n_cols)]
    for i, c in enumerate(cols):
        col_groups[c].append(i)

    # Ordre de lecture des colonnes = x croissant du bord gauche moyen
    col_order = list(range(n_cols))
    def col_xmin(cidx):
        xs = [tokens[i].box.x0 for i in col_groups[cidx]] or [0]
        return sum(xs) / max(1, len(xs))
    col_order.sort(key=col_xmin)

    # RECONSTRUCTION
    paragraphs_all: List[str] = []
    paragraphs_all_no_punct: List[str] = []

    for cidx in col_order:
        idxs = col_groups[cidx]
        if not idxs:
            continue
        # LIGNES
        # on transmet uniquement les tokens de la colonne
        toks_col = [tokens[i] for i in idxs]
        line_groups = group_lines(toks_col, line_tol=args.line_tol)

        # Map local -> global indices
        # mais ici on utilise toks_col directement pour la reconstruction
        lines_text_no_punct: List[str] = []
        for g in line_groups:
            words = [toks_col[j] for j in g]
            line_txt = reconstruct_line(words, space_factor=args.space_factor)
            if line_txt:
                lines_text_no_punct.append(line_txt)

        # PARAGRAPHES
        # Remonte les indices globaux pour calculer gaps verticaux correctement
        # -> on reconstitue les lignes en indices globaux
        global_lines: List[List[int]] = []
        for g in line_groups:
            global_lines.append([ idxs[j] for j in g ])

        paras = lines_to_paragraphs(global_lines, tokens, para_factor=args.para_factor)

        # Texte reconstruit colonne
        start_line = 0
        for para in paras:
            # nombre de lignes dans ce paragraphe
            k = len(para)
            lines = lines_text_no_punct[start_line:start_line + k]
            start_line += k
            para_text_no_punct = "\n".join(lines).strip()
            if not para_text_no_punct:
                continue

            if args.punct:
                para_text = apply_punctuation(para_text_no_punct, model_name=args.punct_model, device=args.device).strip()
            else:
                para_text = para_text_no_punct

            paragraphs_all_no_punct.append(para_text_no_punct)
            paragraphs_all.append(para_text)

    # Sorties
    full_no_punct = "\n\n".join(paragraphs_all_no_punct)
    full = "\n\n".join(paragraphs_all)

    out = {
        "source_image": manifest.get("source_image"),
        "reconstructed": [
            {"paragraph": i, "text_no_punct": paragraphs_all_no_punct[i], "text": paragraphs_all[i]}
            for i in range(len(paragraphs_all))
        ],
        "full_text_no_punct": full_no_punct,
        "full_text": full,
        "columns_detected": n_cols
    }
    out_path = manifest_path.parent / "reconstructed.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== TEXTE RECONSTRUIT (aperçu) ===")
    print(full if args.punct else full_no_punct)
    print(f"\n[reconstruct] Sortie détaillée: {out_path}")


if __name__ == "__main__":
    main()
