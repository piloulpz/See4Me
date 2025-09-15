#!/usr/bin/env python3
import argparse, json, tempfile, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import numpy as np

import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def normalize_box(geom):
    g = np.array(geom, dtype=float)
    if g.ndim == 1 and g.size == 4:
        x0, y0, x1, y1 = g.tolist()
        return float(x0), float(y0), float(x1), float(y1)
    if g.ndim == 2 and g.shape == (2, 2):
        (x0, y0), (x1, y1) = g.tolist()
        return float(x0), float(y0), float(x1), float(y1)
    if g.ndim == 2 and g.shape[1] == 2:
        xs, ys = g[:, 0], g[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    raise ValueError(f"Geometry inattendue: {geom}")

def rel_to_abs(box_rel, W, H, pad=0):
    x0, y0, x1, y1 = box_rel
    x0a = int(round((x0 * W) - pad))
    y0a = int(round((y0 * H) - pad))
    x1a = int(round((x1 * W) + pad))
    y1a = int(round((y1 * H) + pad))
    x0a, y0a = max(0, x0a), max(0, y0a)
    x1a, y1a = min(W, x1a), min(H, y1a)
    return x0a, y0a, x1a, y1a

def crop_region(img: Image.Image, x0, y0, x1, y1):
    return img.crop((x0, y0, x1, y1))

def main():
    ap = argparse.ArgumentParser(description="Segmenter une image en blocs/lignes/mots et enregistrer des crops + manifest.")
    ap.add_argument("image", help="Chemin de l'image source")
    ap.add_argument("--outdir", default="crops", help="Dossier de sortie (défaut: crops)")
    ap.add_argument("--level", choices=["block", "line", "word"], default="line", help="Granularité (défaut: line)")
    ap.add_argument("--rotate", type=int, default=0, help="Rotation horaire en degrés (ex: 90). Défaut: 0")
    ap.add_argument("--pad", type=int, default=0, help="Marge (pixels) autour des boxes (défaut: 0)")
    ap.add_argument("--min_area", type=int, default=0, help="Aire minimale (px) pour garder un crop (0 = pas de filtre)")
    ap.add_argument("--debug", action="store_true", help="Sauve une image debug avec les boxes")
    args = ap.parse_args()

    img_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Charger image, corriger EXIF, convertir RGB, puis rotation HORAIRE si demandé
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if args.rotate % 360 != 0:
        image = image.rotate(-args.rotate, expand=True)  # horaire
    W, H = image.width, image.height

    # Affiche la version réellement utilisée
    #image.show()

    # Device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[segment] device = {device}")

    predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True).to(device)

    # === Sauvegarde TEMPORAIRE de l'image rotatée pour compat Doctr ===
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        image.save(tmp_path)

        # Doctr lit le fichier temp (version rotatée)
        doc = DocumentFile.from_images([tmp_path])
        result = predictor(doc)
        exported = result.export()

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if not exported.get("pages"):
        print("[segment] Aucun contenu détecté.")
        manifest_path = outdir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "source_image": str(img_path),
                "width": W, "height": H,
                "level": args.level, "rotate": args.rotate,
                "pad": args.pad, "min_area": args.min_area,
                "crops": []
            }, f, ensure_ascii=False, indent=2)
        print(f"[segment] 0 crops enregistrés dans: {outdir}")
        print(f"[segment] manifest: {manifest_path}")
        return

    page = exported["pages"][0]

    # Récup zones selon level
    items = []
    for b_idx, block in enumerate(page.get("blocks", [])):
        b_box_rel = normalize_box(block["geometry"])
        if args.level == "block":
            items.append(("block", (b_idx,), b_box_rel))
        for l_idx, line in enumerate(block.get("lines", [])):
            l_box_rel = normalize_box(line["geometry"])
            if args.level == "line":
                items.append(("line", (b_idx, l_idx), l_box_rel))
            for w_idx, word in enumerate(line.get("words", [])):
                w_box_rel = normalize_box(word["geometry"])
                if args.level == "word":
                    items.append(("word", (b_idx, l_idx, w_idx), w_box_rel))

    # Trier haut->bas puis gauche->droite
    def sort_key(t):
        _, _, box_rel = t
        x0, y0, x1, y1 = box_rel
        return (y0, x0)
    items.sort(key=sort_key)

    manifest = {
        "source_image": str(img_path),
        "width": W,
        "height": H,
        "level": args.level,
        "rotate": args.rotate,
        "pad": args.pad,
        "min_area": args.min_area,
        "crops": []
    }

    # Debug draw
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img) if args.debug else None

    kept = 0
    for i, (lvl, idx, box_rel) in enumerate(items):
        x0, y0, x1, y1 = rel_to_abs(box_rel, W, H, pad=args.pad)
        w, h = max(0, x1 - x0), max(0, y1 - y0)
        area = w * h
        if area <= 0:
            continue
        if args.min_area > 0 and area < args.min_area:
            continue

        crop = crop_region(image, x0, y0, x1, y1)
        crop_name = f"{args.level}_{kept:05d}.png"
        crop_path = outdir / crop_name
        crop.save(crop_path)

        manifest["crops"].append({
            "id": kept,
            "level": lvl,
            "index": idx,
            "bbox_rel": [float(b) for b in box_rel],
            "bbox_abs": [int(x0), int(y0), int(x1), int(y1)],
            "path": str(crop_path)
        })
        kept += 1

        if draw:
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)

    # Sauver manifest
    manifest_path = outdir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if args.debug:
        dbg_path = outdir / "debug_boxes.png"
        debug_img.save(dbg_path)
        print(f"[segment] debug: {dbg_path}")

    print(f"[segment] {kept} crops enregistrés dans: {outdir}")
    print(f"[segment] manifest: {manifest_path}")

if __name__ == "__main__":
    main()
