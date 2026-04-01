import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def remove_vignette(image, **kwargs):
    h, w = image.shape[:2]
    # Create circular mask centered at image center
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(h, w) // 2 - 10
    
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply GaussianBlur to mask edges
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    # Multiply image by mask
    mask_float = mask.astype(np.float32) / 255.0
    if len(image.shape) == 3:
        mask_float = np.expand_dims(mask_float, axis=-1)
        
    result = (image.astype(np.float32) * mask_float).clip(0, 255).astype(np.uint8)
    return result

def get_train_transforms(img_size=224):
    return A.Compose([
        A.Lambda(image=remove_vignette, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.RandomShadow(p=0.3),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.2),
        A.MotionBlur(blur_limit=(3, 7), p=0.25),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.ToGray(p=0.05),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.7, 1.0), p=0.5),
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(img_size=224):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
