#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def main():
    ap = argparse.ArgumentParser(description="OCR TrOCR (handwritten) sur un dossier de crops ou un manifest.json")
    ap.add_argument("input", help="Dossier contenant des .png ou chemin vers manifest.json")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--show_first", action="store_true", help="Afficher le premier crop (Aperçu)")
    args = ap.parse_args()

    p = Path(args.input)

    # Charger liste d'images
    crops = []
    manifest_info = {}
    if p.is_file() and p.name == "manifest.json":
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        manifest_info = {
            "source_image": data.get("source_image"),
            "level": data.get("level"),
            "rotate": data.get("rotate"),
        }
        for item in data["crops"]:
            crops.append(Path(item["path"]))
    elif p.is_dir():
        crops = sorted([c for c in p.iterdir() if c.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    else:
        raise SystemExit("Input doit être un dossier de crops OU un manifest.json")

    if not crops:
        raise SystemExit("Aucun crop trouvé.")

    # Device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[trocr] device = {device}")

    # Modèle
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

    results = []
    for i, cp in enumerate(crops):
        img = Image.open(cp).convert("RGB")
        if i == 0 and args.show_first:
            img.show()

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        results.append({"index": i, "path": str(cp), "text": text})

        print(f"[{i+1}/{len(crops)}] {cp.name} -> {text}")

    # Reconstitution naïve : concat ligne par ligne
    full_text = "\n".join([r["text"] for r in results])

    # Sauvegarde
    out_json = (p.parent if p.name == "manifest.json" else p) / "ocr_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "manifest": manifest_info,
            "results": results,
            "full_text": full_text
        }, f, ensure_ascii=False, indent=2)

    print(f"\n=== TEXTE RECONSTITUÉ ===\n{full_text}\n")
    print(f"[trocr] Résultats détaillés : {out_json}")

if __name__ == "__main__":
    main()
