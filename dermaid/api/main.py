import os
import sys
import time
import base64
import numpy as np
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms

# Path setup to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import DermAidModel
from image_quality import check_image_quality
from referral_engine import generate_referral
import config

try:
    # Assuming custom script will implement a `generate_gradcam_overlay(model, img_tensor, original_img)` function
    from gradcam import generate_gradcam_overlay
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


app = FastAPI(title="DermAid API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
model = None
device = None
total_predictions = 0
last_inference_time = None
transform_pipeline = None


@app.on_event("startup")
async def load_model():
    global model, device, transform_pipeline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading DermAidModel on {device}...")
    
    model = DermAidModel()
    
    checkpoint_path = config.CHECKPOINT_DIR / 'dermaid_best.pth'
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Warning: Checkpoint not found. Running with untrained weights for API testing.")
        
    model.to(device)
    model.eval()
    
    transform_pipeline = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    print("Model initialized and ready.")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"{request.method} {request.url.path} - Status: {response.status_code} - Elapsed: {process_time:.4f}s")
    return response


@app.get("/")
def read_root():
    return {
        "status": "running", 
        "model": "DermAid v1.0", 
        "version": "PeaceOfCode2026"
    }


@app.get("/health")
def health_check():
    return {
        "model_loaded": model is not None,
        "last_inference_time": last_inference_time,
        "total_predictions": total_predictions
    }


async def validate_and_read_image(file: UploadFile) -> Image.Image:
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG/PNG allowed.")
        
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        
    try:
        img = Image.open(io.BytesIO(content)).convert('RGB')
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image reading error: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = Form('en')):
    global total_predictions, last_inference_time
    
    # Parse image
    img = await validate_and_read_image(file)
    img_np = np.array(img)
    
    # Run Image Quality Checks
    quality_result = check_image_quality(img_np)
    if not quality_result['is_usable']:
        raise HTTPException(status_code=400, detail=quality_result['message'])
        
    # Inference
    start_t = time.time()
    img_tensor = transform_pipeline(img).unsqueeze(0).to(device)
    result = model.predict(img_tensor)
    
    # Retrieve sorted top-3 conditions and their probability
    cond_probs = result['condition_probs']
    sorted_probs = sorted(enumerate(cond_probs), key=lambda x: x[1], reverse=True)
    top3 = [
        {"condition": config.CLASS_NAMES[i], "probability": round(p * 100, 1)} 
        for i, p in sorted_probs[:3]
    ]
    
    # Process the referral decision engine logic (inc. translations and confidence safety handling)
    predicted_class_name = config.CLASS_NAMES[result['condition_class']]
    severity_tier = config.SEVERITY_MAP[predicted_class_name]
    
    referral = generate_referral(
        condition_code=predicted_class_name,
        severity_tier=severity_tier,
        confidence=result['confidence'],
        top3_conditions=top3,
        lang=lang
    )
    
    # Log telemetry
    infer_ms = (time.time() - start_t) * 1000
    total_predictions += 1
    last_inference_time = time.time()
    
    return {
        "condition": referral['condition'],
        "severity": referral['severity'],
        "urgency_color": referral['urgency_color'],
        "action_title": referral['action_title'],
        "instruction": referral['instruction'],
        "confidence_pct": referral['confidence_pct'],
        "top3_conditions": referral['top3'],
        "inference_ms": round(infer_ms, 2)
    }


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    img = await validate_and_read_image(file)
    
    # Graceful fallback if GradCAM isn't fully loaded yet
    if not GRADCAM_AVAILABLE:
        buffered = io.BytesIO()
        dummy_img = Image.new('RGB', (224, 224), color=(200, 200, 200)) # Gray dummy
        dummy_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"gradcam_base64": img_str, "status": "Mocked (gradcam.py not completely implemented)"}
        
    img_tensor = transform_pipeline(img).unsqueeze(0).to(device)
    
    overlay_img = generate_gradcam_overlay(model, img_tensor, img)
    
    buffered = io.BytesIO()
    overlay_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"gradcam_base64": img_str}
