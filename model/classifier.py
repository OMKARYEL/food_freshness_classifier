from transformers import pipeline
from PIL import Image

classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224"
)

def predict_food_quality(image_path):
    image = Image.open(image_path).convert("RGB")
    results = classifier(image)

    top = results[0]
    score = round(top["score"] * 100, 2)
    label = top["label"].lower()

    if "rotten" in label or "mold" in label or "spoiled" in label:
        quality = "Avoid"
    elif "fresh" in label:
        quality = "Fresh"
    else:
        quality = "Okay"

    return quality, score
