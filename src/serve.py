"""Flask inference service for the CIFAR-10 classifier."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

from model import get_model


app = Flask(__name__)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "checkpoints/classifier_v1.pt",
    )
)

model = None
model_loaded = False

inference_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ]
)


def load_model() -> None:
    """Load the trained model checkpoint into memory."""

    global model, model_loaded

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    architecture = checkpoint.get("architecture", "resnet18")
    num_classes = checkpoint.get("num_classes", len(CLASS_NAMES))

    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_loaded = True


@app.get("/health")
def health():
    """Return service health status."""

    if model_loaded:
        return jsonify(
            {
                "status": "healthy",
                "model_loaded": True,
            }
        ), 200

    return jsonify(
        {
            "status": "unhealthy",
            "model_loaded": False,
        }
    ), 503


@app.post("/predict")
def predict():
    """Run inference on an uploaded image."""

    if not model_loaded:
        return jsonify(
            {
                "error": "Model is not loaded."
            }
        ), 503

    if "image" not in request.files:
        return jsonify(
            {
                "error": "Missing image file. Use form field 'image'."
            }
        ), 400

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return jsonify(
            {
                "error": "No image selected."
            }
        ), 400

    try:
        image = Image.open(uploaded_file.stream).convert("RGB")
        tensor = inference_transform(image)
        tensor = tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = F.softmax(logits, dim=1)[0]

        top_probability, top_index = torch.max(probabilities, dim=0)

        probabilities_list = [
            {
                "class": CLASS_NAMES[index],
                "probability": round(float(probabilities[index]), 6),
            }
            for index in range(len(CLASS_NAMES))
        ]

        return jsonify(
            {
                "predicted_class": CLASS_NAMES[int(top_index)],
                "confidence": round(float(top_probability), 6),
                "probabilities": probabilities_list,
            }
        ), 200

    except Exception as exc:
        return jsonify(
            {
                "error": f"Prediction failed: {exc}"
            }
        ), 400


try:
    load_model()
except Exception as exc:
    print(f"Model loading failed: {exc}", flush=True)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
    )