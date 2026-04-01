import time
import io
import base64
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import config
from model import DermAidModel
from image_quality import check_image_quality
from referral_engine import generate_referral
from uncertainty import predict_with_uncertainty, uncertainty_to_severity_override
from case_logger import CaseLogger

try:
    from gradcam import generate_gradcam_overlay
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

try:
    import tensorflow.lite as tflite
    has_tflite = True
except ImportError:
    has_tflite = False


class DermAidPipeline:
    """
    Primary orchestration engine for DermAid inference, bringing together
    quality checks, clinical model prediction, UI translations, and database logging.
    """
    def __init__(self, model_path='export/dermaid_int8.tflite', use_pytorch=False, device='cpu'):
        self.use_pytorch = use_pytorch
        
        # Determine strict runtime environment compute hardware
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.logger = CaseLogger()
        
        if self.use_pytorch:
            self.model = DermAidModel()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.gradcam_available = GRADCAM_AVAILABLE
            
            self.transform = transforms.Compose([
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
            ])
            print("Pipeline initialized with PyTorch backend.")
        else:
            if not has_tflite:
                try:
                    import tflite_runtime.interpreter as tflite
                except ImportError:
                    raise ImportError("Cannot import TFLite bindings. Please install tensorflow or tflite_runtime.")
            
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.gradcam_available = False
            print("Pipeline initialized with TFLite backend.")

    def predict(self, image_input, lang='en', patient_id=None, worker_id=None,
                phc_name=None, generate_gradcam=False, use_uncertainty=False) -> dict:
        
        start_ms = time.time()
        
        # 1. Image Resolution and Format Standardization
        if isinstance(image_input, str):
            img_pil = Image.open(image_input).convert('RGB')
            img_path = image_input
        elif isinstance(image_input, Image.Image):
            img_pil = image_input.convert('RGB')
            img_path = 'in_memory_PIL'
        elif isinstance(image_input, np.ndarray):
            img_pil = Image.fromarray(image_input).convert('RGB')
            img_path = 'in_memory_ndarray'
        else:
            return {'error': 'Unsupported image format. Must be Path, PIL, or ndarray.', 'quality': None}
            
        img_np = np.array(img_pil)
        
        # 2. Vision Pipeline: Pre-flight Image Quality Analysis
        quality = check_image_quality(img_np)
        if not quality['is_usable']:
            return {'error': quality['message'], 'quality': quality}
            
        auto_escalated = False
        max_uncertainty = 0.0
        
        # 3. Backbone Inference
        if self.use_pytorch:
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            if use_uncertainty:
                unc_res = predict_with_uncertainty(self.model, img_tensor, n_passes=20, device=self.device)
                condition_probs = unc_res['mean_probs']
                condition_class = unc_res['predicted_class']
                max_uncertainty = unc_res['max_uncertainty']
                auto_escalated = unc_res['auto_escalated']
                
                # Still need severity and confidence features directly from the core evaluation
                self.model.eval()
                with torch.no_grad():
                    _, sev_logits, conf_logits = self.model(img_tensor)
                    severity_class = torch.argmax(torch.softmax(sev_logits, dim=1), dim=1).item()
                    confidence = conf_logits.item()
            else:
                self.model.eval()
                res = self.model.predict(img_tensor)
                condition_probs = res['condition_probs']
                condition_class = res['condition_class']
                severity_class = res['severity_class']
                confidence = res['confidence']
                
        else: # TFLite Edge Inference Mode
            resized = img_pil.resize((config.IMG_SIZE, config.IMG_SIZE))
            input_data = np.expand_dims(np.array(resized), axis=0).astype(np.float32)
            
            input_dtype = self.input_details[0]['dtype']
            
            # Accommodate Quantization Parameters Seamlessly
            if input_dtype == np.int8 or input_dtype == np.uint8: 
                input_scale, input_zero_point = self.input_details[0]['quantization']
                if input_scale != 0.0:
                    input_data = (input_data / input_scale + input_zero_point)
                input_data = input_data.astype(input_dtype)
            else:
                 input_data = input_data / 255.0
                
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            outputs = [self.interpreter.get_tensor(od['index']) for od in self.output_details]
            
            cond_probs, sev_probs, conf_val = None, None, 0.0
            
            # Map outputs by expected shapes to decouple from rigid tensor index bindings
            for out in outputs:
                if out.shape == (1, 7): cond_probs = out[0]
                elif out.shape == (1, 3): sev_probs = out[0]
                elif out.shape == (1, 1): conf_val = out[0][0]
                
            # Naive dequantization fallback context (if network didn't output native softmax floats)
            if cond_probs.max() > 1.5: 
                cond_probs = np.exp(cond_probs - np.max(cond_probs)) / np.sum(np.exp(cond_probs - np.max(cond_probs)))
            if sev_probs.max() > 1.5:
                sev_probs = np.exp(sev_probs - np.max(sev_probs)) / np.sum(np.exp(sev_probs - np.max(sev_probs)))
            
            condition_probs = cond_probs.tolist()
            condition_class = int(np.argmax(condition_probs))
            severity_class = int(np.argmax(sev_probs))
            confidence = float(conf_val)
            
        # 4. Synthesize Clinical Features
        condition_code = config.CLASS_NAMES[condition_class]
        
        sorted_probs = sorted(enumerate(condition_probs), key=lambda x: x[1], reverse=True)
        top3 = [{"condition": config.CLASS_NAMES[i], "probability": round(p*100, 1)} for i, p in sorted_probs[:3]]
        
        base_severity = config.SEVERITY_MAP[condition_code]
        severity_tier = uncertainty_to_severity_override(base_severity, max_uncertainty) if auto_escalated else base_severity
        
        # 5. Connect to Multilingual Action Engine
        referral = generate_referral(condition_code, severity_tier, confidence, top3, lang=lang)
        
        # 6. Optional Explainability Overlay
        gradcam_b64 = None
        if generate_gradcam and self.use_pytorch and self.gradcam_available:
            try:
                overlay = generate_gradcam_overlay(self.model, img_tensor, img_pil)
                buffered = io.BytesIO()
                overlay.save(buffered, format="PNG")
                gradcam_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"Warning: GradCAM generation failed: {e}")
                
        inference_ms = round((time.time() - start_ms) * 1000, 2)
        
        # 7. Local SQLLite Logging
        case_id = None
        if patient_id:
            db_payload = {
                'condition_code': condition_code,
                'condition': referral['condition'],
                'severity': severity_tier,
                'urgency_color': referral['urgency_color'],
                'confidence_pct': round(confidence * 100, 1),
                'action_title': referral['action_title'],
                'auto_escalated': auto_escalated,
                'max_uncertainty': max_uncertainty
            }
            case_id = self.logger.log_case(
                patient_id=patient_id, 
                worker_id=worker_id, 
                phc_name=phc_name, 
                image_path=img_path, 
                result_dict=db_payload, 
                lang=lang
            )
            
        return {
            'quality_check': quality,
            'condition_code': condition_code,
            'condition_name': referral['condition'],
            'condition_probs': condition_probs,
            'severity_tier': severity_tier,
            'urgency_color': referral['urgency_color'],
            'action_title': referral['action_title'],
            'instruction': referral['instruction'],
            'confidence_pct': round(confidence * 100, 1),
            'top3': top3,
            'inference_ms': inference_ms,
            'auto_escalated': auto_escalated,
            'gradcam_overlay': gradcam_b64,
            'case_id': case_id
        }
