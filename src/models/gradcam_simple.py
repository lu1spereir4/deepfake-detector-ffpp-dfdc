"""
Grad-CAM para ResNet18 - Detección de Deepfakes
Versión simplificada para análisis de 5 imágenes específicas
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        # Obtener predicción (2 clases: 0=REAL, 1=FAKE)
        probs = F.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        prediction = "REAL" if pred_idx == 0 else "FAKE"
        confidence = probs[0, pred_idx].item()
        
        # Clase objetivo
        if target_class is None:
            target_class = pred_idx
        
        # Backward para obtener gradientes
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Generar CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        cam = torch.sum(weights.view(-1, 1, 1) * activations, dim=0)  # [H, W]
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), prediction, confidence


def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def visualize_gradcam(model_path, image_path, output_dir, target_class=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar modelo
    model = load_model(model_path)
    target_layer = model.layer4[-1]
    
    # Cargar imagen
    image = Image.open(image_path).convert('RGB')
    
    # Preprocesamiento (DEBE coincidir con el entrenamiento)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    # Obtener predicción ANTES de registrar hooks
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        pred_label = "REAL" if prediction == 0 else "FAKE"
        confidence = probs[0, prediction].item()
    
    # Crear Grad-CAM
    torch.set_grad_enabled(True)
    gradcam = GradCAM(model, target_layer)
    cam, _, _ = gradcam.generate_cam(input_tensor, target_class)
    
    # Generar heatmap
    cam_resized = cv2.resize(cam, (image.width, image.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superponer
    image_np = np.array(image)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    
    # Determinar título según análisis
    if target_class is None:
        title = f"Grad-CAM: {pred_label} ({confidence:.2%})"
    elif target_class == 0:
        title = f"Análisis REAL forzado\n(Predicción real: {pred_label} {confidence:.2%})"
    else:
        title = f"Análisis FAKE forzado\n(Predicción real: {pred_label} {confidence:.2%})"
    
    # Guardar imagen con título
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradcam_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'prediction': pred_label,
        'confidence': confidence
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--target_class', type=int, default=None)
    args = parser.parse_args()
    
    result = visualize_gradcam(args.model_path, args.image_path, args.output_dir, args.target_class)
    print(f"Predicción: {result['prediction']} ({result['confidence']:.2%})")
