python3 segment_blocks.py new_data/img_xyz.png --outdir crops_img --level word   # ou line
python3 trocr_batch.py crops_img/manifest.json
python3 reconstruct_sentences.py crops_img --punct
