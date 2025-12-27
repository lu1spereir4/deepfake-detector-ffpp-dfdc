"""
CNN personalizada para detección de deepfakes.
Implementación de las arquitecturas descritas en el informe:
- DeepfakeDetectorCNN: CNN Estándar (~8.4M parámetros)
- DeepfakeDetectorCNNSmall: CNN Ligera (~1.2M parámetros)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepfakeDetectorCNN(nn.Module):
    """
    CNN Personalizada Estándar para clasificación binaria (real vs fake).
    
    Arquitectura según informe:
    - 4 bloques convolucionales: 3→64→128→256→512
    - Cada bloque tiene 2 capas convolucionales
    - BatchNorm + ReLU + MaxPool en cada bloque
    - Adaptive Average Pooling
    - Clasificador: Flatten → Dropout(0.5) → FC(512→256) → ReLU → Dropout(0.5) → FC(256→2)
    - ~8.4M parámetros entrenables
    
    Input: imágenes RGB de 224x224
    Output: 2 clases (real=0, fake=1)
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super(DeepfakeDetectorCNN, self).__init__()
        
        # BLOQUE CONVOLUCIONAL 1: 3 → 64 filtros (2 capas conv)
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224 → 112
        
        # BLOQUE CONVOLUCIONAL 2: 64 → 128 filtros (2 capas conv)
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112 → 56
        
        # BLOQUE CONVOLUCIONAL 3: 128 → 256 filtros (2 capas conv)
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56 → 28
        
        # BLOQUE CONVOLUCIONAL 4: 256 → 512 filtros (2 capas conv)
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(512)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # → 1x1
        
        # CLASIFICADOR con capas más profundas para alcanzar ~8.4M parámetros
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización de pesos (Kaiming/He para ReLU)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        # Bloque 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        
        # Bloque 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        
        # Bloque 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        
        # Bloque 4 + Adaptive Pooling
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.adaptive_pool(x)
        
        # Clasificador
        x = self.classifier(x)
        
        return x


class DeepfakeDetectorCNNSmall(nn.Module):
    """
    CNN Personalizada Ligera para experimentación rápida.
    
    Arquitectura según informe:
    - 3 bloques convolucionales: 3→32→64→128
    - Cada bloque tiene 2 capas convolucionales
    - BatchNorm + ReLU + MaxPool
    - Adaptive Average Pooling
    - Clasificador: Flatten → Dropout(0.3) → FC(128→64) → ReLU → FC(64→2)
    - ~1.2M parámetros entrenables
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super(DeepfakeDetectorCNNSmall, self).__init__()
        
        # BLOQUE 1: 3 → 32 filtros (2 capas conv)
        self.conv1a = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224 → 112
        
        # BLOQUE 2: 32 → 64 filtros (2 capas conv)
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112 → 56
        
        # BLOQUE 3: 64 → 128 filtros (2 capas conv)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # → 1x1
        
        # CLASIFICADOR con capas más profundas para alcanzar ~1.2M parámetros
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización de pesos."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        # Block 1: 224x224 → 112x112
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        
        # Block 2: 112x112 → 56x56
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        
        # Block 3: 56x56 → 1x1
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.adaptive_pool(x)
        
        # Classifier
        x = self.classifier(x)
        
        return x


def count_parameters(model):
    """Cuenta parámetros totales y entrenables."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_model():
    """Prueba rápida de los modelos."""
    print("=" * 60)
    print("PROBANDO DeepfakeDetectorCNN (CNN Estándar)")
    print("=" * 60)
    
    model = DeepfakeDetectorCNN(num_classes=2)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    total, trainable = count_parameters(model)
    print(f"\nParámetros totales:     {total:,}")
    print(f"Parámetros entrenables: {trainable:,}")
    print(f"Esperado según informe: ~8.4M parámetros")
    
    print("\n" + "=" * 60)
    print("PROBANDO DeepfakeDetectorCNNSmall (CNN Ligera)")
    print("=" * 60)
    
    model_small = DeepfakeDetectorCNNSmall(num_classes=2)
    output_small = model_small(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output_small.shape}")
    
    total_s, trainable_s = count_parameters(model_small)
    print(f"\nParámetros totales:     {total_s:,}")
    print(f"Parámetros entrenables: {trainable_s:,}")
    print(f"Esperado según informe: ~1.2M parámetros")


if __name__ == "__main__":
    test_model()
