# See4Me

🚀 Description

This project implements a complete OCR pipeline:

  - Segmentation of an image into blocks/lines/words with [DocTR]

  - Text recognition with [TrOCR (base handwritten)]

  - Post-editing with a fine-tuned Flan-T5 model (ouiyam/see4me-flan5-rewriter-base) to produce clean English text, optionally with a title

  - Reformulation into a lesson-style Markdown document (summary paragraph + bullet points) using TinyLlama

The main script main.py automatically runs the entire pipeline.

⚙️ Install
1. Clone the project
  git clone https://github.com/piloulpz/See4Me 
  cd See4Me

2. Create a virtual environment
  python3 -m venv .venv
  source .venv/bin/activate   # Linux / macOS
  .\.venv\Scripts\activate    # Windows PowerShell

3. Install dependencies
  pip install --upgrade pip
  pip install torch torchvision torchaudio
  pip install transformers datasets
  pip install python-doctr[torch] pillow numpy


⚠️ On Mac M1/M2, install torch with Metal (MPS) support:

  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

▶️ Usage
Simple command
python3 main.py new_data/img_test_x.png


By default, this will:
  - Segment the image
  - Run TrOCR
  - Post-edit with your fine-tuned Flan-T5 model
  - Generate a title
  - Produce a lesson Markdown file

Main options
  --llm_model : checkpoint for post-editing (default: ouiyam/see4me-flan5-rewriter-base)
  --llm_tokens : max tokens for post-editing (default 160)
  --title / --no-title : enable/disable title generation
  --title_model : different model for title (default = same as --llm_model)
  --title_max_new_tokens : max tokens for title (default 16)
  --trocr_tokens : max tokens for TrOCR (default 128)
  --seg_level : segmentation granularity (block, line, word)
  --seg_pad : padding around crops (px)
  --seg_min_area : minimum crop size filter
  --seg_rotate : image rotation (°)
  --reform_outdir : output folder for reformulated lessons (default out)

Full example
  python3 main.py new_data/img_test_9.png \
    --llm_model ouiyam/see4me-flan5-rewriter-base \
    --llm_tokens 160 \
    --title \
    --reform_outdir lessons_out

📂 Results

After execution, you will find:

In crops_<image_name>:

  - manifest.json : crop metadata
  - ocr_results.json : raw TrOCR output
  - reconstructed.json : post-edited text + title
  - text_final.txt : clean text only
  - title.txt : generated title

In the reformulation output folder (default out/):

<image_name>_course.md : Markdown lesson (title + summary paragraph + key points)
