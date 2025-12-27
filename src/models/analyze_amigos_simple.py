"""
Análisis Grad-CAM de 5 imágenes específicas
Versión simplificada
"""
import os
from gradcam_simple import visualize_gradcam
import matplotlib.pyplot as plt
from PIL import Image

# Configuración
MODEL_PATH = '../../models/deepfake_detector_resnet18_ffpp.pth'
IMAGES_DIR = '../../data/processed/ffpp/test/real'
OUTPUT_DIR = '../../results/gradcam_amigos'
AMIGOS = ['amigo1.jpeg', 'amigo2.jpeg', 'amigo3.jpeg', 'amigo4.jpeg', 'amigo5.jpeg']

print("="*60)
print("ANÁLISIS GRAD-CAM: IMÁGENES DE AMIGOS")
print("="*60)

results = []

for amigo in AMIGOS:
    image_path = os.path.join(IMAGES_DIR, amigo)
    
    if not os.path.exists(image_path):
        print(f"⚠️ {amigo} no encontrada")
        continue
    
    print(f"\nProcesando: {amigo}")
    output_subdir = os.path.join(OUTPUT_DIR, amigo.replace('.jpeg', ''))
    
    try:
        # Análisis principal
        result = visualize_gradcam(MODEL_PATH, image_path, output_subdir)
        
        # Análisis forzado REAL
        visualize_gradcam(MODEL_PATH, image_path, 
                         os.path.join(output_subdir, 'forced_real'), 
                         target_class=0)
        
        # Análisis forzado FAKE
        visualize_gradcam(MODEL_PATH, image_path, 
                         os.path.join(output_subdir, 'forced_fake'), 
                         target_class=1)
        
        results.append({
            'image': amigo,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'path': output_subdir
        })
        
        print(f"  ✅ {result['prediction']} ({result['confidence']:.2%})")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Resumen
print(f"\n{'='*60}")
print("RESUMEN")
print(f"{'='*60}")
for r in results:
    print(f"{r['image']:15s} → {r['prediction']:5s} ({r['confidence']:.2%})")

# Crear comparación visual
if len(results) >= 2:
    print(f"\nGenerando comparación visual...")
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 5*len(results)))
    
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, r in enumerate(results):
        # Cargar las 3 visualizaciones
        main_img = Image.open(os.path.join(r['path'], 'gradcam_result.png'))
        real_img = Image.open(os.path.join(r['path'], 'forced_real', 'gradcam_result.png'))
        fake_img = Image.open(os.path.join(r['path'], 'forced_fake', 'gradcam_result.png'))
        
        axes[idx, 0].imshow(main_img)
        axes[idx, 0].set_title(f"{r['image']}\nPredicción", fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(real_img)
        axes[idx, 1].set_title("Forzado REAL", fontweight='bold')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(fake_img)
        axes[idx, 2].set_title("Forzado FAKE", fontweight='bold')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, 'comparacion_completa.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Guardado en: {comparison_path}")

print(f"\n✅ Análisis completado. Resultados en: {OUTPUT_DIR}")
