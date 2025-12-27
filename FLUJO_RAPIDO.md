# GUÍA RÁPIDA DE FLUJO - Detector de Deepfakes

## 🎯 Objetivo del Proyecto
Detectar si una imagen de rostro es REAL o FAKE usando ResNet18 + Grad-CAM para explicabilidad.

---

## 📊 FLUJO COMPLETO (De datos a producción)

### PASO 1: Preprocesamiento de Datos
**Archivo:** `src/data/extract_and_crop_ffpp.py`

**Qué hace:**
- Lee 7,000 videos del dataset FaceForensics++ (1,000 reales + 6,000 fake)
- Extrae 10 frames uniformemente de cada video
- Detecta rostros con MTCNN (confianza >95%)
- Recorta rostros y guarda como imágenes
- **Output:** 47,994 imágenes en `data/processed/ffpp/`

**Comando:**
```bash
python src/data/extract_and_crop_ffpp.py
```

**Resultado:**
```
data/processed/ffpp/
├── train/    # 33,595 imágenes
├── val/      # 7,200 imágenes
└── test/     # 7,199 imágenes
```

---

### PASO 2: Generar Metadata
**Archivo:** `src/data/index_ffpp.py`

**Qué hace:**
- Escanea todas las imágenes procesadas
- Genera CSV con: ruta, split (train/val/test), etiqueta (real/fake)
- **Output:** `data/processed/ffpp/ffpp_images_metadata.csv`

**Comando:**
```bash
python src/data/index_ffpp.py
```

---

### PASO 3: Entrenamiento del Modelo
**Archivo:** `src/models/train_resnet_ffpp.py`

**Qué hace:**
- Carga ResNet18 preentrenada en ImageNet
- Reemplaza última capa: 1000 clases → 2 clases (real/fake)
- Entrena 10 épocas con:
  - Ponderación de clases (4.83 real, 1.00 fake)
  - Augmentation: RandomCrop, Flip, ColorJitter
  - Adam optimizer (lr=1e-4)
- Guarda mejor modelo según validation accuracy
- **Output:** `models/deepfake_detector_resnet18_ffpp.pth`

**Comando:**
```bash
python src/models/train_resnet_ffpp.py
```

**Variables clave para cambiar:**
```python
MODEL_TYPE = "resnet18"  # o "custom", "custom_small"
batch_size = 64
num_epochs = 10
lr = 1e-4
```

**Output esperado:**
```
Época 01/10 | Train loss: 0.5234, acc: 0.7845 | Val loss: 0.4321, acc: 0.8123
...
Época 08/10 | Train loss: 0.2981, acc: 0.8934 | Val loss: 0.3913, acc: 0.8683
  🔥 Nuevo mejor modelo guardado (val_acc: 0.8683)
```

---

### PASO 4: Evaluación Detallada
**Archivo:** `src/models/evaluate_resnet_ffpp.py`

**Qué hace:**
- Carga modelo entrenado
- Evalúa en conjunto de test (7,199 imágenes)
- Calcula matriz de confusión
- Muestra métricas por clase (precision, recall, F1)
- **Output:** Métricas en terminal

**Comando:**
```bash
python src/models/evaluate_resnet_ffpp.py
```

**Output esperado:**
```
Test Accuracy: 88.67%
Test Loss: 0.3441

Matriz de Confusión:
              Predicción
           REAL    FAKE
REAL        983     158    (Recall: 86.15%)
FAKE        658    5400    (Recall: 89.14%)

Precision REAL: 59.90%
Precision FAKE: 97.16%
```

---

### PASO 5: Predicción en Nueva Imagen
**Archivo:** `src/models/predict_image.py`

**Qué hace:**
- Carga modelo entrenado
- Procesa imagen (resize 224×224, normalización)
- Predice clase (REAL/FAKE) y confianza
- **Output:** Predicción en terminal

**Comando:**
```bash
python src/models/predict_image.py ruta/a/imagen.jpg
```

**Output esperado:**
```
📸 Imagen: amigo2.jpeg
🤖 Predicción: REAL
📊 Confianza: 96.52%
```

---

### PASO 6: Grad-CAM (Explicabilidad)
**Archivo:** `src/models/analyze_amigos_simple.py`

**Qué hace:**
- Carga 5 imágenes de prueba
- Aplica Grad-CAM en capa layer4 de ResNet18
- Genera 3 visualizaciones por imagen:
  1. Predicción normal
  2. Análisis forzado REAL
  3. Análisis forzado FAKE
- Crea grid comparativo de los 5 casos
- **Output:** Visualizaciones en `results/gradcam_amigos/`

**Comando:**
```bash
cd src/models
python analyze_amigos_simple.py
```

**Output esperado:**
```
Analizando amigo1.jpeg...
  Predicción: REAL (68.51%)
  ✓ Guardado en results/gradcam_amigos/amigo1/

...

✓ Grid completo guardado en results/gradcam_amigos/comparison_grid.png
```

---

### PASO 7: API REST (Despliegue)
**Archivo:** `src/api/main.py`

**Qué hace:**
- Levanta servidor FastAPI
- Endpoint `/predict` para subir imagen y obtener predicción
- Documentación automática en `/docs`
- **Output:** API REST en puerto 8000

**Comando:**
```bash
cd src/api
uvicorn main:app --reload
```

**Uso:**
```bash
# Abrir en navegador
http://localhost:8000/docs

# O usar curl
curl -X POST "http://localhost:8000/predict" \
  -F "file=@imagen.jpg"
```

**Output esperado:**
```json
{
  "prediction": "REAL",
  "confidence": 0.9652,
  "probabilities": {
    "real": 0.9652,
    "fake": 0.0348
  }
}
```

---

## 🔑 ARCHIVOS CLAVE A ENTENDER

### 1. `src/datasets/ffpp_faces.py` (DataLoader)
**Función:** Lee CSV metadata y carga imágenes con transformaciones PyTorch

**Código esencial:**
```python
class FFPPFacesDataset(Dataset):
    def __init__(self, csv_path, split='train', transform=None):
        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split]  # Filtrar train/val/test
        
    def __getitem__(self, idx):
        img = Image.open(img_path)
        label = 0 if label_str == 'real' else 1  # 0=REAL, 1=FAKE
        return self.transform(img), label
```

---

### 2. `src/models/custom_cnn.py` (Arquitectura personalizada)
**Función:** Define 2 CNNs desde cero (8.49M y 1.11M parámetros)

**Arquitectura CNN Estándar:**
- 4 bloques convolucionales (64→128→256→512 filtros)
- BatchNorm + ReLU + MaxPool
- Adaptive Average Pooling
- Clasificador: Dropout → FC(512→256) → Dropout → FC(256→2)

---

### 3. `src/models/gradcam_simple.py` (Explicabilidad)
**Función:** Implementa Grad-CAM para visualizar atención del modelo

**Concepto:**
1. Forward pass → obtiene activaciones de capa objetivo
2. Backward pass → calcula gradientes respecto a clase predicha
3. Promedio ponderado: `CAM = ReLU(Σ(gradientes × activaciones))`
4. Resize CAM a tamaño de imagen y overlay con heatmap

**Código esencial:**
```python
# 1. Forward
output = model(input_tensor)
pred_idx = torch.argmax(output)

# 2. Backward
output[0, pred_idx].backward()

# 3. Grad-CAM
weights = gradients.mean(dim=(2, 3))  # Global average pooling
cam = torch.sum(weights[:, :, None, None] * activations, dim=1)
cam = F.relu(cam)  # ReLU para quedarse con activaciones positivas
```

---

## 📈 CONCEPTOS CLAVE

### Transfer Learning (ResNet18)
- **Idea:** Usar pesos preentrenados en ImageNet (1.2M imágenes)
- **Ventaja:** Modelo ya sabe extraer features visuales (bordes, texturas, formas)
- **Ajuste:** Solo reemplazar última capa y entrenar con nuestro dataset

### Ponderación de Clases
- **Problema:** Dataset desbalanceado (84% fake, 16% real)
- **Solución:** Asignar mayor peso a clase minoritaria en loss function
  ```python
  weights = torch.tensor([4.83, 1.00])  # [REAL, FAKE]
  criterion = nn.CrossEntropyLoss(weight=weights)
  ```
- **Efecto:** Modelo no colapsa a predecir siempre "fake"

### Grad-CAM
- **Problema:** ¿En qué partes de la imagen se fija el modelo?
- **Solución:** Visualizar activaciones de última capa convolucional
- **Interpretación:** Zonas rojas/amarillas = regiones importantes para decisión

---

## 🎓 ORDEN RECOMENDADO PARA ESTUDIAR

1. **Entender el flujo de datos:**
   - `extract_and_crop_ffpp.py` → `index_ffpp.py` → `ffpp_faces.py`

2. **Entender el entrenamiento:**
   - `train_resnet_ffpp.py` (ver bucle de entrenamiento y validación)

3. **Entender la evaluación:**
   - `evaluate_resnet_ffpp.py` (ver cálculo de métricas)

4. **Entender Grad-CAM:**
   - `gradcam_simple.py` (ver hooks y generación de CAM)

5. **Entender el despliegue:**
   - `src/api/main.py` (ver endpoints FastAPI)

---

## 🚀 COMANDOS RÁPIDOS

```bash
# Flujo completo desde cero
python src/data/extract_and_crop_ffpp.py   # ~2-3 horas
python src/data/index_ffpp.py              # ~1 min
python src/models/train_resnet_ffpp.py     # ~40 min (CPU)
python src/models/evaluate_resnet_ffpp.py  # ~2 min
cd src/models && python analyze_amigos_simple.py  # ~30 seg

# Solo predicción con modelo ya entrenado
python src/models/predict_image.py imagen.jpg

# API REST
cd src/api && uvicorn main:app --reload
```

---

## 📊 RESULTADOS ESPERADOS

- **Accuracy en test:** 88.67%
- **Recall clase REAL:** 86.15%
- **Recall clase FAKE:** 89.14%
- **Precision clase FAKE:** 97.16% (muy confiable cuando predice fake)
- **Tiempo de inferencia:** ~100ms por imagen (CPU)
- **Validación externa:** 100% en 5 fotos reales de amigos

---

**Autor:** Luis - Universidad del Bío-Bío  
**Fecha:** Diciembre 2025
