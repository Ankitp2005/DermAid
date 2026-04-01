import albumentations as A
import cv2
import numpy as np


def remove_vignette(image, **kwargs):
    """Remove the dark circular border typical of dermoscopy images."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2 - 10, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    image = (image * mask[:, :, np.newaxis]).astype(np.uint8)
    return image


# Domain-adaptation transforms: simulate smartphone conditions
# applied ON TOP of dermoscopy images during training
DOMAIN_ADAPT_TRANSFORMS = A.Compose(
    [
        # 1. Remove dermoscopy artifacts (dark circular vignette)
        A.Lambda(image=remove_vignette, p=0.5),
        # 2. Simulate smartphone camera noise
        A.GaussNoise(var_limit=(10, 50), p=0.4),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        # 3. Simulate real-world lighting variation
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.RandomShadow(p=0.3),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.2),
        # 4. Simulate motion blur / shaky hand
        A.MotionBlur(blur_limit=(3, 7), p=0.25),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # 5. Color shift (smartphone white balance variation)
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5
        ),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        # 6. Simulate Indian skin tone range (Fitzpatrick IV-VI)
        # Darken toward brown undertones
        A.ToGray(p=0.05),  # occasional grayscale for robustness
        # 7. Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
        # 8. Simulate zoom / crop (worker won't frame perfectly)
        A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0), p=0.5),
        # Final resize for MobileNetV3
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Basic transforms for validation/testing (no domain adaptation)
BASIC_TRANSFORMS = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Training transforms that combine domain adaptation with basic preprocessing
def get_train_transforms():
    return DOMAIN_ADAPT_TRANSFORMS


def get_val_transforms():
    return BASIC_TRANSFORMS


def get_test_transforms():
    return BASIC_TRANSFORMS
