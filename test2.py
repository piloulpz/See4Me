#!/usr/bin/env python3

import os, sys
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ========= Config =========
LANGS = ['en', 'fr']                           # langues pour la détection (influence peu les boîtes)
MODEL_NAME = "microsoft/trocr-base-printed"    # "printed" souvent mieux pour écriture au feutre; sinon "...-handwritten"
MAX_LENGTH = 128
MIN_BOX_AREA = 250                              # filtre les minuscules boîtes
MAX_SIDE = 2200                                 # downscale si image énorme

# ========= Device (MPS sur Mac si dispo) =========
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ========= Détection (boîtes) via EasyOCR =========
def detect_boxes(image_path: str) -> list[tuple[int,int,int,int]]:
    """
    Returns list of axis-aligned boxes (x1,y1,x2,y2) sorted in reading order.
    Robust to EasyOCR's different output shapes.
    """
    import easyocr, numpy as np

    reader = easyocr.Reader(LANGS, gpu=False)

    # 1) Try fast detector
    det = reader.detect(image_path, width_ths=0.7, mag_ratio=1.5)
    boxes = []

    def quad_to_xyxy(q):
        """q can be [[x,y],...]*4  OR np.array shape (4,2)  OR flat len=8  OR [x1,y1,x2,y2]."""
        a = np.array(q)
        if a.ndim == 1:
            if len(a) == 8:               # flat quad
                a = a.reshape(4, 2)
            elif len(a) == 4:             # already xyxy
                x1, y1, x2, y2 = map(int, a)
                return x1, y1, x2, y2
            else:
                raise ValueError(f"Unrecognized box shape 1D len={len(a)}: {a}")
        if a.shape == (4, 2):             # quad
            x1, y1 = int(a[:, 0].min()), int(a[:, 1].min())
            x2, y2 = int(a[:, 0].max()), int(a[:, 1].max())
            return x1, y1, x2, y2
        raise ValueError(f"Unrecognized box shape {a.shape}: {a}")

    try:
        if det and det[0]:
            quads = det[0][0]  # list of quads
            for q in quads:
                try:
                    x1, y1, x2, y2 = quad_to_xyxy(q)
                    if (x2 - x1) * (y2 - y1) >= MIN_BOX_AREA:  # filter tiny boxes
                        boxes.append((x1, y1, x2, y2))
                except Exception:
                    continue
    except Exception:
        pass

    # 2) Fallback: use readtext (slower but stable)
    if not boxes:
        results = reader.readtext(image_path, detail=1, paragraph=False)
        # results: list of (box, text, conf), where box is 4-point polygon
        for poly, _, _ in results:
            try:
                x1, y1, x2, y2 = quad_to_xyxy(poly)
                if (x2 - x1) * (y2 - y1) >= MIN_BOX_AREA:
                    boxes.append((x1, y1, x2, y2))
            except Exception:
                continue

    # reading order
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


# ========= OCR TrOCR =========
class Trocr:
    def __init__(self, model_name: str, device: str):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        self.device = device

    @torch.inference_mode()
    def ocr_pil(self, img: Image.Image) -> str:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        ids = self.model.generate(**inputs, max_length=MAX_LENGTH)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ========= Prétraitement doux pour le crop =========
def preprocess_crop_for_trocr(crop_bgr: np.ndarray) -> Image.Image:
    # gris + égalisation + petit renforcement
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    # réinjecte dans RGB pour TrOCR
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

# ========= Pipeline =========
def run(image_path: str):
    # charge et éventuel downscale (accélère la détection)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise SystemExit(f"Image introuvable: {image_path}")
    h, w = img_bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        r = MAX_SIDE / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w*r), int(h*r)), interpolation=cv2.INTER_AREA)

    # sauvegarde temporaire (EasyOCR détecte depuis un fichier)
    tmp_path = "__tmp_detect.jpg"
    cv2.imwrite(tmp_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    boxes = detect_boxes(tmp_path)
    os.remove(tmp_path)

    # overlay pour debug
    overlay = img_bgr.copy()
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite("overlay_boxes.jpg", overlay)

    # TrOCR
    ocr = Trocr(MODEL_NAME, DEVICE)

    lines = []
    os.makedirs("crops", exist_ok=True)
    for i, (x1,y1,x2,y2) in enumerate(boxes, 1):
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0: 
            continue
        cv2.imwrite(f"crops/crop_{i:03d}.jpg", crop)
        pil = preprocess_crop_for_trocr(crop)
        text = ocr.ocr_pil(pil)
        if text:
            lines.append((y1, x1, text))

    # tri final + export
    lines.sort(key=lambda t: (t[0], t[1]))
    with open("out.md", "w", encoding="utf-8") as f:
        for _,__,txt in lines:
            f.write(txt + "\n")

    print(f"[OK] {len(lines)} lignes reconnues")
    print(" - out.md (texte)")
    print(" - overlay_boxes.jpg (boîtes)")
    print(" - dossier crops/ (crops utilisés)")

# ========= Main =========
if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "image_test.jpeg"
    run(img_path)
