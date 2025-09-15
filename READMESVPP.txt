python3 segment_blocks.py new_data/img_xyz.png --outdir crops_img --level word   # ou line
python3 trocr_batch.py crops_img/manifest.json
python3 reconstruct_sentences.py crops_img --punct<


python3 segment_blocks.py new_data/img_test_9.png --outdir crops_img --level line --pad 8 --min_area 1500
python3 trocr_batch.py crops_img/manifest.json --max_new_tokens 64
python3 reconstruct_sentences.py crops_img --punct --line_tol 0.7 --space_factor 0.35 --para_factor 1.5

"You are an expert copyeditor. Fix the following OCR text in English.\n"
    "Requirements:\n"
    "- Preserve ALL content and the ORIGINAL line breaks.\n"
    "- Return the SAME number of lines; edit each line independently.\n"
    "- Correct spelling, grammar, casing, spacing, and punctuation.\n"
    "- Remove OCR artifacts (e.g., stray letters, duplicated tokens, 0→o, 1→l, '0 0' noise, bad hyphenation).\n"
    "- Do NOT translate, omit, merge, reorder, or summarize.\n"
    "- You can delete all the words that are OCR fails and you have to make sure that all the sentences are correct.\n"
    "Answer with ONLY the corrected text.\n\n"
    "Text:\n"