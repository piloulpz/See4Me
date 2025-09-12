from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from PIL import Image
import torch

# ---- Device: MPS (Apple Silicon) -> CUDA -> CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ---- Donnée: IAM line (grayscale) -> converti en RGB
ds = load_dataset("Teklia/IAM-line", split="test")
image: Image.Image = ds[0]["image"].convert("RGB")
image.show()   # ouvre l’image dans Aperçu sur macOS


# ---- Modèle
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

# ---- Inference
inputs = processor(images=image, return_tensors="pt").to(device)
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=64)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n=== TEXTE RECONNU ===")
print(text)
