#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd: str):
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main():
    ap = argparse.ArgumentParser(
        description="Pipeline complet: segmentation -> TrOCR -> reconstruction (LLM post-edit fine-tuné) -> reformulation (cours)"
    )
    ap.add_argument("image", help="Chemin de l'image à traiter (ex: new_data/img_test_9.png)")

    # === LLM post-edit (aligné au fine-tune) ===
    ap.add_argument(
        "--llm_model",
        default="ouiyam/see4me-flan5-rewriter-base",
        help="Checkpoint seq2seq pour la post-édition (HF Hub: ouiyam/see4me-flan5-rewriter-base)",
    )
    ap.add_argument("--llm_tokens", type=int, default=160,
                    help="max_new_tokens pour la post-édition (160 conseillé avec le fine-tune)")
    ap.add_argument("--title", action="store_true", default=True,
                    help="Générer un titre (par défaut ON). Utilise --no-title pour désactiver.")
    ap.add_argument("--no-title", dest="title", action="store_false")
    ap.add_argument("--title_model", default=None,
                    help="Modèle pour le titre (défaut: même que --llm_model)")
    ap.add_argument("--title_max_new_tokens", type=int, default=16,
                    help="max_new_tokens pour le titre (défaut: 16)")

    # === TrOCR ===
    ap.add_argument("--trocr_tokens", type=int, default=128,
                    help="max_new_tokens pour TrOCR")

    # === Segmentation ===
    ap.add_argument("--seg_level", choices=["block", "line", "word"], default="line")
    ap.add_argument("--seg_pad", type=int, default=8)
    ap.add_argument("--seg_min_area", type=int, default=1500)
    ap.add_argument("--seg_rotate", type=int, default=0)

    # === Reformulation (cours) ===
    ap.add_argument("--reform_outdir", default="out",
                    help="Répertoire de sortie pour le cours reformulé (défaut: out)")

    args = ap.parse_args()

    image = Path(args.image)
    if not image.exists():
        sys.exit(f"Erreur: fichier introuvable {image}")

    outdir = f"crops_{image.stem}"

    # Étape 1 : segmentation
    run_cmd(
        f"python3 segment_blocks.py {image} "
        f"--outdir {outdir} "
        f"--level {args.seg_level} --pad {args.seg_pad} "
        f"--min_area {args.seg_min_area} --rotate {args.seg_rotate}"
    )

    # Étape 2 : OCR avec TrOCR
    run_cmd(
        f"python3 trocr_batch.py {outdir}/manifest.json --max_new_tokens {args.trocr_tokens}"
    )

    # Étape 3 : Reconstruction avec ton modèle fine-tuné (Hub)
    cmd_recon = (
        f"python3 reconstruct_sentences.py {outdir} "
        f"--fix_model {args.llm_model} "
        f"--max_new_tokens {args.llm_tokens}"
    )
    if args.title:
        cmd_recon += " --title"
        if args.title_model:
            cmd_recon += f" --title_model {args.title_model}"
        if args.title_max_new_tokens is not None:
            cmd_recon += f" --title_max_new_tokens {args.title_max_new_tokens}"
    run_cmd(cmd_recon)

    # Étape 4 : Reformulation en cours structuré
    course_outdir = Path(args.reform_outdir)
    course_outdir.mkdir(parents=True, exist_ok=True)
    course_md = course_outdir / f"{image.stem}_course.md"

    run_cmd(
        f"python3 reformulation.py --input {outdir} --out {course_md}"
    )
    run_cmd(f"cat {course_md}")

    print(f"\n[main] Terminé. Résultats dans: {outdir} (reconstructed.json, text_final.txt, title.txt) + {course_md}")

if __name__ == "__main__":
    main()
