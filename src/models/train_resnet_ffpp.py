from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.datasets.ffpp_faces import FFPPFacesDataset
from src.models.custom_cnn import DeepfakeDetectorCNN, DeepfakeDetectorCNNSmall


DATA_META_CSV = Path("data/processed/ffpp/ffpp_images_metadata.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# opciones: "custom", "custom_small", "resnet18"
MODEL_TYPE = "resnet18"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(batch_size: int = 64):

    # Transformaciones (augmentations simples)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = FFPPFacesDataset(
        csv_path=DATA_META_CSV,
        split="train",
        transform=train_transform,
    )

    val_dataset = FFPPFacesDataset(
        csv_path=DATA_META_CSV,
        split="val",
        transform=eval_transform,
    )

    test_dataset = FFPPFacesDataset(
        csv_path=DATA_META_CSV,
        split="test",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def build_model(num_classes: int = 2, model_type: str = MODEL_TYPE):
    """
    Construye el modelo según el tipo especificado.
    
    Args:
        num_classes: Número de clases de salida (2 para real/fake)
        model_type: "custom", "custom_small", o "resnet18"
    
    Returns:
        modelo PyTorch
    """
    if model_type == "custom":
        # Tu CNN personalizada (control total)
        model = DeepfakeDetectorCNN(num_classes=num_classes, dropout_rate=0.5)
        print(f"✅ Usando modelo CUSTOM CNN")
        
    elif model_type == "custom_small":
        # Versión pequeña para pruebas rápidas
        model = DeepfakeDetectorCNNSmall(num_classes=num_classes, dropout_rate=0.3)
        print(f"✅ Usando modelo CUSTOM CNN (Small)")
        
    elif model_type == "resnet18":
        # ResNet18 pre-entrenada (transfer learning)
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        print(f"✅ Usando ResNet18 pre-entrenada")
        
    else:
        raise ValueError(f"model_type no válido: {model_type}")
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parámetros totales: {total_params:,}")
    print(f"   Parámetros entrenables: {trainable_params:,}")
    
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    device = get_device()
    print(f"🖥️  Usando device: {device}\n")

    # Configuración según informe
    batch_size = 64
    num_epochs = 10
    lr = 1e-4
    weight_decay = 1e-5

    print("=" * 70)
    print(f"ENTRENANDO MODELO: {MODEL_TYPE.upper()}")
    print("=" * 70)
    print(f"📊 Configuración:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Épocas: {num_epochs}")
    print(f"   - Learning rate: {lr}")
    print(f"   - Weight decay: {weight_decay}")
    print()

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = build_model(num_classes=2, model_type=MODEL_TYPE).to(device)

    # ==================================================================
    # PONDERACIÓN DE CLASES según el informe
    # Conteos del dataset de entrenamiento:
    # - Real: 5,760
    # - Fake: 27,835
    # Total: 33,595
    # 
    # Pesos calculados: w_class = N / (2 * N_class)
    # - w_real = 33,595 / (2 * 5,760) = 2.92
    # - w_fake = 33,595 / (2 * 27,835) = 0.60
    # ==================================================================
    class_counts = [5760, 27835]  # [real, fake]
    total_samples = sum(class_counts)
    
    # Calcular pesos según fórmula del informe
    weights = torch.tensor([
        total_samples / (2 * class_counts[0]),  # peso para clase real
        total_samples / (2 * class_counts[1])   # peso para clase fake
    ], dtype=torch.float32)
    
    print(f"⚖️  Ponderación de clases (para manejar desbalance):")
    print(f"   - Clase Real: {class_counts[0]:,} muestras → peso {weights[0]:.2f}")
    print(f"   - Clase Fake: {class_counts[1]:,} muestras → peso {weights[1]:.2f}")
    print()

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    
    # Nombre del modelo basado en el tipo
    model_filename = f"deepfake_detector_{MODEL_TYPE}_ffpp.pth"
    best_model_path = MODEL_DIR / model_filename

    print("🚀 Iniciando entrenamiento...\n")

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start

        print(
            f"Época {epoch:02d}/{num_epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Guardar mejor modelo según accuracy en validación
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                    "model_type": MODEL_TYPE,
                },
                best_model_path,
            )
            print(f"  🔥 Nuevo mejor modelo guardado (val_acc: {best_val_acc:.4f})")

    print("\n" + "=" * 70)
    print("✅ Entrenamiento completado")
    print("=" * 70)

    # Cargar mejor modelo y evaluar en test
    print(f"\n📂 Cargando mejor modelo desde: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("🧪 Evaluando en conjunto de prueba (test)...\n")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print("=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)
    print(f"📈 Mejor val_acc: {best_val_acc:.4f} (época {checkpoint['epoch']})")
    print(f"📊 Test accuracy: {test_acc:.4f}")
    print(f"📊 Test loss:     {test_loss:.4f}")
    print(f"💾 Modelo guardado en: {best_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
