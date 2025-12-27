"""
Script para evaluar modelos entrenados en el conjunto de test.
"""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.datasets.ffpp_faces import FFPPFacesDataset
from src.models.custom_cnn import DeepfakeDetectorCNN, DeepfakeDetectorCNNSmall


DATA_META_CSV = Path("data/processed/ffpp/ffpp_images_metadata.csv")
MODEL_DIR = Path("models")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_type: str, checkpoint_path: Path, device):
    """Carga el modelo desde checkpoint."""
    
    if model_type == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_type == "custom":
        model = DeepfakeDetectorCNN(num_classes=2)
    elif model_type == "custom_small":
        model = DeepfakeDetectorCNNSmall(num_classes=2)
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate(model, dataloader, criterion, device):
    """Evalúa el modelo en un conjunto de datos."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def get_test_dataloader(batch_size: int = 64):
    """Crea el dataloader de test."""
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = FFPPFacesDataset(DATA_META_CSV, split='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    return test_loader


def main():
    device = get_device()
    print(f"🖥️  Usando device: {device}\n")
    
    # Modelos a evaluar
    models_to_eval = [
        ("resnet18", MODEL_DIR / "deepfake_detector_resnet18_ffpp.pth"),
        ("custom", MODEL_DIR / "deepfake_detector_custom_ffpp.pth"),
        ("custom_small", MODEL_DIR / "deepfake_detector_custom_small_ffpp.pth"),
    ]
    
    test_loader = get_test_dataloader()
    criterion = nn.CrossEntropyLoss()
    
    print("=" * 80)
    print("EVALUACIÓN DE MODELOS EN CONJUNTO DE TEST")
    print("=" * 80)
    
    results = []
    
    for model_type, checkpoint_path in models_to_eval:
        if not checkpoint_path.exists():
            print(f"⚠️  Modelo no encontrado: {checkpoint_path}")
            continue
        
        print(f"\n📊 Evaluando modelo: {model_type.upper()}")
        print(f"   Checkpoint: {checkpoint_path}")
        
        model, checkpoint = load_model(model_type, checkpoint_path, device)
        
        # Información del checkpoint
        if 'epoch' in checkpoint:
            print(f"   Época guardada: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            print(f"   Val accuracy (guardado): {checkpoint['val_acc']:.4f}")
        
        # Evaluar en test
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
        
        print(f"   📈 Test Loss: {test_loss:.4f}")
        print(f"   📈 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        results.append({
            'model': model_type,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'val_acc': checkpoint.get('val_acc', 0)
        })
    
    # Resumen comparativo
    print("\n" + "=" * 80)
    print("RESUMEN COMPARATIVO")
    print("=" * 80)
    print(f"{'Modelo':<20} {'Val Acc':<12} {'Test Acc':<12} {'Test Loss':<12}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x['test_acc'], reverse=True):
        print(f"{r['model']:<20} {r['val_acc']:<12.2%} {r['test_acc']:<12.2%} {r['test_loss']:<12.4f}")
    
    print("=" * 80)
    
    # Mejor modelo
    best = max(results, key=lambda x: x['test_acc'])
    print(f"\n🏆 Mejor modelo: {best['model'].upper()}")
    print(f"   Test Accuracy: {best['test_acc']:.2%}")
    print(f"   Test Loss: {best['test_loss']:.4f}")


if __name__ == "__main__":
    main()
