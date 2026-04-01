import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

import config

class GradCAMModelWrapper(torch.nn.Module):
    """
    Wraps DermAidModel so that it outputs only the condition logits exactly as expected
    by the pytorch-grad-cam library.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        cond_logits, _, _ = self.model(x)
        return cond_logits


class DermAidGradCAM:
    def __init__(self, model):
        self.wrapped_model = GradCAMModelWrapper(model)
        self.wrapped_model.eval()
        
        # Target the final convolution block of the MobileNetV3 backbone
        target_layer = self.wrapped_model.model.backbone.features[-1]
        
        self.cam = GradCAM(model=self.wrapped_model, target_layers=[target_layer])
        
    def generate(self, image_tensor, target_class=None):
        if target_class is None:
            # None implies we generate the CAM for the highest scoring predicted class
            targets = None
        else:
            targets = [ClassifierOutputTarget(target_class)]
            
        # Returns [batch_size, width, height] numpy float array (0-1)
        grayscale_cam = self.cam(input_tensor=image_tensor, targets=targets)
        return grayscale_cam[0, :]
        
    def overlay(self, image_np, grayscale_cam, alpha=0.4):
        # Convert image_np to 0.0-1.0 float required by pytorch-grad-cam
        float_img = image_np.astype(np.float32) / 255.0
        
        visualization = show_cam_on_image(
            float_img, 
            grayscale_cam, 
            use_rgb=True, 
            colormap=cv2.COLORMAP_JET, 
            image_weight=alpha
        )
        return visualization
        
    def generate_all_classes(self, image_tensor, image_np):
        overlays = {}
        for idx, class_name in enumerate(config.CLASS_NAMES):
            grayscale = self.generate(image_tensor, target_class=idx)
            overlay_img = self.overlay(image_np, grayscale)
            overlays[class_name] = overlay_img
        return overlays


def save_gradcam_figure(image_np, overlays_dict, predicted_class, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Original Image in first slot
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    # Draw all 7 heatmaps
    for i, class_name in enumerate(config.CLASS_NAMES):
        ax = axes[i + 1]
        ax.imshow(overlays_dict[class_name])
        ax.axis('off')
        
        if class_name == predicted_class:
            # Highlight the predicted class with a thick red border
            rect = plt.Rectangle((0, 0), image_np.shape[1]-1, image_np.shape[0]-1, 
                                 fill=False, edgecolor='red', linewidth=6)
            ax.add_patch(rect)
            ax.set_title(f"{class_name.upper()} Heatmap\n(Predicted)", 
                         fontsize=14, color='red', weight='bold')
        else:
            ax.set_title(f"{class_name.upper()} Heatmap", fontsize=14)
            
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"GradCAM grid saved to {save_path}")


def generate_gradcam_overlay(model, image_tensor, image_pil):
    """
    Convenience wrapper used by FastAPI endpoint.
    Retrieves the CAM of the highest predicted class.
    """
    image_pil_resized = image_pil.resize((config.IMG_SIZE, config.IMG_SIZE))
    image_np = np.array(image_pil_resized)
    
    gradcam = DermAidGradCAM(model)
    grayscale = gradcam.generate(image_tensor, target_class=None)
    overlay = gradcam.overlay(image_np, grayscale)
    
    return Image.fromarray(overlay)
