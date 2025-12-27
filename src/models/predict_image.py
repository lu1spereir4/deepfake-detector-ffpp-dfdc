import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image


# --- Configuración global ---
BASE_DIR = Path(__file__).resolve().parents[2]
CKPT_PATH = BASE_DIR / "models" / "deepfake_detector_resnet18_ffpp.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model() -> nn.Module:
    # Igual que en entrenamiento: ResNet18 con 2 clases
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


# Transforms igual que en test
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@torch.no_grad()
def predict_image(image_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    # Cargar imagen
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # shape (1, 3, H, W)

    # Modelo
    model = build_model()
    model = load_model(model, CKPT_PATH)

    # Forward
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Etiquetas: asumiendo 0=real, 1=fake
    classes = ["real", "fake"]
    pred_idx = int(probs.argmax())
    pred_label = classes[pred_idx]
    prob_pred = float(probs[pred_idx])

    return {
        "pred_label": pred_label,
        "pred_idx": pred_idx,
        "prob_real": float(probs[0]),
        "prob_fake": float(probs[1]),
        "prob_pred": prob_pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Ruta a la imagen a clasificar")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    result = predict_image(image_path)

    print(f"Imagen: {image_path}")
    print(f"Predicción: {result['pred_label']} (idx={result['pred_idx']})")
    print(f"Prob. real: {result['prob_real']:.4f}")
    print(f"Prob. fake: {result['prob_fake']:.4f}")


if __name__ == "__main__":
    main()
