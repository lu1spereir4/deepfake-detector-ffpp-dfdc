from pathlib import Path
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms


app = FastAPI(title="Deepfake Detector API")

BASE_DIR = Path(__file__).resolve().parents[2]
CKPT_PATH = BASE_DIR / "models" / "deepfake_detector_resnet18_ffpp.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model() -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    model.eval()
    return model


def load_model(model: nn.Module, ckpt_path: Path) -> nn.Module:
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Cargamos el modelo UNA VEZ al iniciar el servidor
model = build_model()
model = load_model(model, CKPT_PATH)


@app.get("/")
def read_root():
    return {
        "message": "Deepfake Detector API - Universidad del Bío-Bío",
        "endpoints": {
            "health": "/health",
            "predict": "/predict/",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
def health_check():
    """Endpoint de health check para verificar que la API está funcionando."""
    return {
        "status": "ok",
        "model": "resnet18_ffpp",
        "device": str(device)
    }


@app.post("/predict/")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """
    Predice si una imagen de rostro es real o deepfake.
    
    Args:
        file: Imagen a analizar (JPG, PNG, etc.)
    
    Returns:
        JSON con predicción, confianza y probabilidades
    """
    # Validar tipo de archivo
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer la imagen: {str(e)}")

    # Preprocesar
    x = transform(image).unsqueeze(0).to(device)

    # Inferencia
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    classes = ["real", "fake"]
    pred_idx = int(probs.argmax())
    pred_label = classes[pred_idx]

    result = {
        "prediction": pred_label,
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            "real": float(probs[0]),
            "fake": float(probs[1])
        },
        "model": "resnet18_ffpp"
    }

    return JSONResponse(content=result)
