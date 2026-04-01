import cv2
import numpy as np

def check_skin_pixels(image: np.ndarray) -> bool:
    """
    Checks if an image contains a significant amount of skin pixels.
    Input image is expected to be an RGB numpy array.
    """
    # Convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # In OpenCV, Hue is 0-179 instead of 0-359.
    # So 0-30 degrees -> 0-15; 340-360 degrees -> 170-179.
    # Saturation: 15-170
    # Value: 60-255
    
    # Mask 1 (Hue 0-15)
    lower_skin1 = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin1 = np.array([15, 170, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    
    # Mask 2 (Hue 170-179)
    lower_skin2 = np.array([170, 15, 60], dtype=np.uint8)
    upper_skin2 = np.array([179, 170, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    
    # Combine masks
    skin_mask = cv2.bitwise_or(mask1, mask2)
    
    # Calculate percentage of skin pixels
    total_pixels = hsv.shape[0] * hsv.shape[1]
    skin_pixels = np.sum(skin_mask > 0)
    skin_percentage = skin_pixels / total_pixels
    
    # Return true if more than 20% of pixels fall in the skin tone range
    return skin_percentage > 0.20


def check_image_quality(image: np.ndarray) -> dict:
    """
    Evaluates image quality based on blur, brightness, contrast, and skin tone prescence.
    Input image is expected to be an RGB numpy array.
    """
    # Convert image to grayscale for intensity-based metrics
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute metrics
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    contrast = gray.std()
    has_skin = check_skin_pixels(image)
    
    # Evaluate boolean flags
    is_blurry = blur_score < 100
    too_dark = brightness < 40
    too_bright = brightness > 220
    low_contrast = contrast < 20
    no_skin_detected = not has_skin
    
    is_usable = not (is_blurry or too_dark or too_bright or low_contrast or no_skin_detected)
    
    # Determine the first failing message
    if is_blurry:
        message = 'Image too blurry. Move closer and hold camera steady.'
    elif too_dark:
        message = 'Image too dark. Move to brighter area or use flash.'
    elif too_bright:
        message = 'Image overexposed. Avoid direct flash or sunlight.'
    elif low_contrast:
        message = 'Low contrast. Ensure lesion is clearly visible.'
    elif no_skin_detected:
        message = 'No skin detected. Photograph the affected skin area directly.'
    else:
        message = 'Image quality: Good. Analyzing...'
        
    # Calculate a composite 0-100 quality score 
    # (heuristically derived from blur and contrast)
    normalized_blur = min(100.0, blur_score / 3.0)
    normalized_contrast = min(100.0, contrast * 1.5)
    skin_bonus = 100.0 if has_skin else 0.0
    
    base_score = (normalized_blur * 0.4) + (normalized_contrast * 0.4) + (skin_bonus * 0.2)
    
    # Deduct points for poor lighting
    if too_dark or too_bright:
        base_score -= 30.0
        
    quality_score = max(0.0, min(100.0, base_score))

    return {
        'is_blurry': is_blurry,
        'too_dark': too_dark,
        'too_bright': too_bright,
        'low_contrast': low_contrast,
        'no_skin_detected': no_skin_detected,
        'is_usable': is_usable,
        'blur_score': round(float(blur_score), 2),
        'brightness': round(float(brightness), 2),
        'contrast': round(float(contrast), 2),
        'quality_score': round(float(quality_score), 2),
        'message': message
    }
