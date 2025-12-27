import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
import numpy as np

from src.datasets.ffpp_faces import FFPPFacesDataset  # importa tu dataset


def build_model(device: torch.device) -> nn.Module:
    # Igual que en el entrenamiento: ResNet18 con 2 clases
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)
    return model


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Soporta tanto dict con 'model_state_dict' como state_dict directo
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
    return model


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    # Matriz de confusión 2x2: filas = etiqueta real [0=real, 1=fake],
    # columnas = predicción [0=real, 1=fake]
    conf_mat = np.zeros((2, 2), dtype=int)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

            # Actualizar matriz de confusión
            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_mat[int(t), int(p)] += 1

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc, conf_mat


def print_metrics(conf_mat: np.ndarray):
    # Métricas por clase
    for cls_idx, cls_name in enumerate(["real (0)", "fake (1)"]):
        tp = conf_mat[cls_idx, cls_idx]
        fp = conf_mat[:, cls_idx].sum() - tp
        fn = conf_mat[cls_idx, :].sum() - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"\nClase {cls_name}:")
        print(f"  TP = {tp}, FP = {fp}, FN = {fn}")
        print(f"  Precision = {precision:.4f}")
        print(f"  Recall    = {recall:.4f}")
        print(f"  F1-score  = {f1:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando device:", device)

    BASE_DIR = Path(__file__).resolve().parents[2]
    METADATA_CSV = BASE_DIR / "data" / "processed" / "ffpp" / "ffpp_images_metadata.csv"
    CKPT_PATH = BASE_DIR / "models" / "deepfake_detector_resnet18_ffpp.pth"

    # Transforms iguales a test en el entrenamiento
    img_size = 224
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ⚠️ IMPORTANTE: instanciar igual que en train_resnet_ffpp.py
    test_dataset = FFPPFacesDataset(
        csv_path=METADATA_CSV,
        split="test",
        transform=test_transform,
        # si en train_resnet_ffpp.py usan otros argumentos (p.ej. base_dir=...),
        # agrégalos aquí igual
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"[FFPPFacesDataset] test: {len(test_dataset)} imágenes")

    # Modelo
    model = build_model(device)
    model = load_checkpoint(model, CKPT_PATH, device)

    # Evaluar
    test_loss, test_acc, conf_mat = evaluate(model, test_loader, device)

    print("\n===== Evaluación en TEST =====")
    print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
    print("\nMatriz de confusión (filas = real, columnas = pred):")
    print(conf_mat)

    print_metrics(conf_mat)


if __name__ == "__main__":
    main()
