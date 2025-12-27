# Detector de Deepfakes - FaceForensics++

Sistema de detección de deepfakes con ResNet18 + Grad-CAM.

## Estructura del Proyecto

```
├── data/                      # Datos procesados (47,994 imágenes)
├── models/                    # Modelos entrenados (.pth)
├── results/                   # Resultados de análisis
├── figs/                      # Figuras para el informe
├── src/
│   ├── data/                  # Scripts de preprocesamiento
│   │   ├── extract_and_crop_ffpp.py   # Extrae frames y detecta rostros (MTCNN)
│   │   └── index_ffpp.py              # Genera metadata CSV
│   ├── datasets/              # Dataset PyTorch
│   │   └── ffpp_faces.py              # DataLoader para imágenes
│   ├── models/                # Modelos y entrenamiento
│   │   ├── custom_cnn.py              # CNN personalizada (8.49M params)
│   │   ├── train_resnet_ffpp.py       # Entrenamiento ResNet18
│   │   ├── evaluate_model.py          # Evaluación comparativa
│   │   ├── evaluate_resnet_ffpp.py    # Evaluación ResNet18 detallada
│   │   ├── predict_image.py           # Predicción individual
│   │   ├── gradcam_simple.py          # Implementación Grad-CAM
│   │   └── analyze_amigos_simple.py   # Análisis Grad-CAM (5 fotos)
│   └── api/
│       └── main.py                     # API REST (FastAPI)
├── informe.tex                # Informe LaTeX completo
└── requirements.txt           # Dependencias Python

```

## Flujo de Ejecución

### 1. Preprocesamiento (primera vez)
```bash
# Extrae frames y detecta rostros con MTCNN
python src/data/extract_and_crop_ffpp.py

# Genera metadata CSV
python src/data/index_ffpp.py
```

### 2. Entrenamiento
```bash
# Entrenar ResNet18 (mejor modelo: 88.67% test accuracy)
python src/models/train_resnet_ffpp.py
```

### 3. Evaluación
```bash
# Evaluación detallada en test
python src/models/evaluate_resnet_ffpp.py

# Comparación de los 3 modelos
python src/models/evaluate_model.py
```

### 4. Predicción
```bash
# Predicción individual
python src/models/predict_image.py path/to/image.jpg

# Demo con 10 imágenes incluidas
python test_demo.py
```

**Imágenes de prueba incluidas:**
- `data/processed/ffpp/test/real/amigo1-5.jpeg` (fotos reales)
- `data/processed/ffpp/test/fake/test_fake_1-5.jpg` (deepfakes)

### 5. Grad-CAM (Explicabilidad)
```bash
cd src/models
python analyze_amigos_simple.py
```

### 6. API REST
```bash
cd src/api
uvicorn main:app --reload
# Abrir: http://localhost:8000/docs
```

### 7. Docker
```bash
docker-compose up
```
Acceder a: http://localhost:8000/docs

## Resultados Clave

| Modelo | Params | Val Acc | Test Acc |
|--------|--------|---------|----------|
| ResNet18 | 11.18M | 86.83% | **88.67%** |
| CNN Estándar | 8.49M | 83.19% | 81.92% |
| CNN Ligera | 1.11M | 73.74% | 72.12% |

- **Mejor modelo:** ResNet18 con transfer learning
- **Dataset:** FaceForensics++ desde Kaggle (7,000 videos → 47,994 imágenes)
- **Técnicas fake:** Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter
- **Compresión:** c23 (H.264 CRF 23)
- **Validación externa:** 100% accuracy en 5 fotos reales

## Dependencias Principales

- Python 3.11
- PyTorch 2.5.1
- torchvision 0.20.1
- OpenCV 4.10.0
- facenet-pytorch 2.6.0 (MTCNN)
- FastAPI 0.115.6
- matplotlib, seaborn

## Dataset

**Fuente:** [Kaggle - FaceForensics++ C23](https://www.kaggle.com/datasets/xdxd003/ff-c23)

Estructura esperada:
```
data/raw/ffpp/
├── original/        # 1,000 videos reales
├── Deepfakes/       # 1,000 videos fake
├── Face2Face/       # 1,000 videos fake
├── FaceSwap/        # 1,000 videos fake
├── NeuralTextures/  # 1,000 videos fake
├── FaceShifter/     # 1,000 videos fake
└── csv/             # Metadatos
```

## Configuración de Entrenamiento

- **Batch size:** 64
- **Épocas:** 10
- **Learning rate:** 1e-4
- **Optimizer:** Adam con weight decay 1e-5
- **Loss:** CrossEntropyLoss con ponderación de clases (4.83 real, 1.00 fake)
- **Augmentation:** RandomResizedCrop, HorizontalFlip, ColorJitter
- **Hardware:** GPU (Para los entrenamientos se uso una Nvidia GeForce 4070 RTX)

## Autor

Luis Pereira Toledo - Universidad del Bío-Bío  
Ingeniería Civil en Informática  
Asignatura: Introduccion a las aplicaciones de algoritmos de Machine Learning y Deep Learning  
Diciembre 2025


## 📄 Technical Report
The full project report (SoftwareX format) is available in:

docs/PS_IAMLDL_Pereira_Luis.pdf