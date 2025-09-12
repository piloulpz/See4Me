# test.py
# Usage:
#   python3 test.py path/to/image.jpg [handwritten|printed]
# Par défaut: handwritten (marqueur). Langue: anglais.

import sys, os
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ----------------------------
# Configs
# ----------------------------
MODEL_KIND = sys.argv[2] if len(sys.argv) > 2 else "handwritten"
if MODEL_KIND not in {"handwritten", "printed"}:
    MODEL_KIND = "handwritten"

MODEL_NAME = (
    "microsoft/trocr-large-handwritten" if MODEL_KIND == "handwritten"
    else "microsoft/trocr-large-printed"
)

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "image_test.jpeg"

# Chargement du modèle
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def to_pil(arr_bgr):
    return Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))

def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([[0, 0],[maxW - 1, 0],[maxW - 1, maxH - 1],[0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH), flags=cv2.INTER_CUBIC)
    return warped

# ----------------------------
# 1) Détection & redressement du tableau
# ----------------------------
def detect_and_warp_board(bgr):
    h, w = bgr.shape[:2]
    scale = 1000.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        small = bgr.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 80, 200)
    edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_area = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best = approx.reshape(-1,2); best_area = area

    if best is not None:
        if scale < 1.0:
            best = best / scale
        return four_point_transform(bgr, best.astype(np.float32))
    return bgr

# ----------------------------
# 2) Nettoyage illumination & reflets
# ----------------------------
def normalize_illumination(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51,51))
    background = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel)
    vn = cv2.subtract(v, background)
    vn = cv2.normalize(vn, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    hsvn = cv2.merge([h, s, vn])
    return cv2.cvtColor(hsvn, cv2.COLOR_HSV2BGR)

def reduce_glare(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresh_val = np.percentile(gray, 99.0)
    glare = (gray >= thresh_val).astype(np.uint8)*255
    glare = cv2.dilate(glare, np.ones((5,5), np.uint8), iterations=1)
    return cv2.inpaint(bgr, glare, 5, cv2.INPAINT_TELEA)

# ----------------------------
# 3) Extraction encre & segmentation en lignes
# ----------------------------
def ink_mask_and_clean(gray):
    g = cv2.GaussianBlur(gray, (3,3), 0)
    g = cv2.equalizeHist(g)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th

def segment_lines(ink_mask):
    h, w = ink_mask.shape
    hist = ink_mask.sum(axis=1) / 255
    lines = []
    in_line = False; start = 0
    thresh = max(10, 0.01 * w)
    for y in range(h):
        if not in_line and hist[y] > thresh:
            in_line = True; start = y
        elif in_line and hist[y] <= thresh:
            end = y
            if end - start > 6:
                s = max(0, start-2); e = min(h, end+2)
                lines.append((s, e))
            in_line = False
    if in_line:
        e = h-1
        if e - start > 6:
            lines.append((max(0, start-2), e))
    return lines

# ----------------------------
# 4) OCR
# ----------------------------
@torch.inference_mode()
def ocr_pil(pil_img, max_len=256):
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(
        pixel_values,
        max_length=max_len,
        num_beams=8,
        early_stopping=True,
        length_penalty=1.1,
        no_repeat_ngram_size=4,
    )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def best_orientation_text(crop_bgr):
    cands = [
        crop_bgr,
        cv2.rotate(crop_bgr, cv2.ROTATE_180),
        cv2.rotate(crop_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(crop_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    best_text, best_score = "", -1
    for c in cands:
        t = ocr_pil(to_pil(c))
        s = sum(ch.isalnum() or ch in " -_.,;:!?()[]'\"" for ch in t)
        if s > best_score:
            best_text, best_score = t, s
    return best_text, best_score

# ----------------------------
# 5) Pipeline complet
# ----------------------------
def run(image_path):
    if not os.path.exists(image_path):
        print(f"Image introuvable: {image_path}")
        return

    bgr = cv2.imread(image_path)
    if bgr is None:
        print("Échec de lecture de l'image.")
        return

    board = detect_and_warp_board(bgr)
    board = normalize_illumination(board)
    board = reduce_glare(board)

    min_chan = np.min(board, axis=2).astype(np.uint8)
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    gray = cv2.min(gray, min_chan)

    ink = ink_mask_and_clean(gray)
    lines = segment_lines(ink)

    results = []
    for (y0, y1) in lines:
        line_crop = board[y0:y1, :]
        lh = y1 - y0
        target_h = 56
        scale = max(1.0, target_h / max(1, lh))
        if scale > 1.01:
            line_crop = cv2.resize(
                line_crop,
                (int(line_crop.shape[1]*scale), int(line_crop.shape[0]*scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        text, _ = best_orientation_text(line_crop)
        if text:
            results.append(text)

    if results:
        print("\n=== TEXTE RECONNU (whiteboard, {}): ===".format(MODEL_KIND))
        for t in results:
            print(t)
    else:
        print("[Info] Lignes non détectées. Tentative en bloc…")
        pil_full = to_pil(board)
        text = ocr_pil(pil_full, max_len=512)
        print("\n=== TEXTE RECONNU (bloc) ===")
        print(text)

if __name__ == "__main__":
    run(IMG_PATH)
