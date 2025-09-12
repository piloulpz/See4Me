import sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# ---- Vérifier argument
if len(sys.argv) < 2:
    print("Usage: python trocr_local.py <fichier_image>")
    sys.exit(1)

image_path = sys.argv[1]

# ---- Device: MPS (Apple Silicon) -> CUDA -> CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ---- Charger image locale + rotation
try:
    image = Image.open(image_path).convert("RGB")
    image = image.rotate(-90, expand=True)  # ✅ rotation horaire 90°
except Exception as e:
    print(f"Erreur ouverture image {image_path}: {e}")
    sys.exit(1)

image.show()  # ouvre dans Aperçu

# ---- Modèle
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

# ---- Inference
inputs = processor(images=image, return_tensors="pt").to(device)
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=128)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n=== TEXTE RECONNU ===")
print(text)
