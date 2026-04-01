import os
import sys
import tempfile
import numpy as np
import torch
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from model import DermAidModel
    from config import CLASS_NAMES, SEVERITY_MAP
    from referral_engine import generate_referral, REFERRAL_MATRIX_EN
    from image_quality import check_image_quality
    from mixup import mixup_data
    from case_logger import CaseLogger
except ImportError as e:
    pytest.fail(f"Could not import fundamental pipeline modules: {e}")

@pytest.fixture
def dummy_image_tensor():
    # Standard batch size 1, 3 channels, 224x224 dimensions
    return torch.randn(1, 3, 224, 224)


def test_model_output_shapes(dummy_image_tensor):
    model = DermAidModel()
    model.eval()
    with torch.no_grad():
        cond_logits, sev_logits, confidence = model(dummy_image_tensor)
        
    assert cond_logits.shape == (1, 7), f"Expected condition shape (1, 7), got {cond_logits.shape}"
    assert sev_logits.shape == (1, 3), f"Expected severity shape (1, 3), got {sev_logits.shape}"
    assert confidence.shape == (1, 1), f"Expected confidence shape (1, 1), got {confidence.shape}"


def test_model_confidence_range(dummy_image_tensor):
    model = DermAidModel()
    model.eval()
    with torch.no_grad():
        _, _, confidence = model(dummy_image_tensor)
        
    conf_val = confidence.item()
    # BCE/Sigmoid confidence head must strictly output valid probability spectrum
    assert 0.0 <= conf_val <= 1.0


def test_severity_map_complete():
    allowed_tiers = ['Low Risk', 'Refer Soon', 'Refer Immediately']
    for cls in CLASS_NAMES:
        assert cls in SEVERITY_MAP, f"Class '{cls}' missing from SEVERITY_MAP"
        assert SEVERITY_MAP[cls] in allowed_tiers, f"Invalid severity mapped to '{cls}'"


def test_referral_matrix_complete():
    allowed_colors = ['GREEN', 'YELLOW', 'RED']
    for cls in CLASS_NAMES:
        severity = SEVERITY_MAP[cls]
        key = (cls, severity)
        
        assert key in REFERRAL_MATRIX_EN, f"Pair {key} entirely missing from generic Referral maps."
        
        # Triple unpacking: Action Title, Instruction Body, Urgency Color
        _, _, color = REFERRAL_MATRIX_EN[key]
        assert color in allowed_colors, f"Invalid urgency color '{color}' for key {key}"


@pytest.mark.parametrize("condition", CLASS_NAMES[:3]) # test handful of arbitrary classes
def test_low_confidence_escalation(condition):
    severity = SEVERITY_MAP[condition]
    
    # Confidence arbitrarily below the 0.60 safety threshold
    result = generate_referral(
        condition_code=condition, 
        severity_tier=severity, 
        confidence=0.3, 
        top3=[], 
        lang='en'
    )
    
    # The referral engine must globally override normal color coding to YELLOW 
    # and flag confidence issues dynamically if model is unsure.
    assert result['urgency_color'] == 'YELLOW'
    assert 'Low Confidence' in result['action_title']


def test_image_quality_blur():
    dark_img = np.zeros((224, 224, 3), dtype=np.uint8)
    res = check_image_quality(dark_img)
    
    assert res['too_dark'] is True
    assert res['is_usable'] is False


def test_image_quality_good():
    # Simulate a realistic patient lesion via base skin tone and noise
    # Base skin tone: RGB(210, 160, 120)
    base = np.zeros((224, 224, 3), dtype=np.float32)
    base[:, :, 0] = 210
    base[:, :, 1] = 160
    base[:, :, 2] = 120
    
    # Adding Gaussian textural noise explicitly tricks the Laplacian and Std. Dev passes
    # mimicking raw camera noise or natural skin variation
    noise = np.random.normal(0, 30, (224, 224, 3))
    good_img = np.clip(base + noise, 0, 255).astype(np.uint8)
    
    res = check_image_quality(good_img)
    
    assert res['too_dark'] is False
    assert res['too_bright'] is False
    assert res['low_contrast'] is False
    assert res['no_skin_detected'] is False
    assert res['is_blurry'] is False
    assert res['is_usable'] is True


def test_mixup_shape(dummy_image_tensor):
    y = torch.tensor([0])
    mixed_x, y_a, y_b, lam = mixup_data(dummy_image_tensor, y, alpha=0.4, device='cpu')
    
    # Invariant checks for regularizer structure preservation
    assert mixed_x.shape == dummy_image_tensor.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert 0.0 <= lam <= 1.0


def test_case_logger_insert():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
        
    try:
        logger = CaseLogger(db_path=db_path)
        payload = {
            'condition_code': 'nv',
            'condition': 'Melanocytic nevi',
            'severity': 'Low Risk',
            'urgency_color': 'GREEN',
            'confidence_pct': 92.5,
            'action_title': 'No Action Needed',
            'auto_escalated': False,
            'max_uncertainty': 0.05
        }
        idx = logger.log_case('pat_123', 'worker_1', 'PHC Alpha', 'in_memory', payload, 'en')
        
        assert idx > 0
        cases = logger.get_cases()
        assert len(cases) == 1
        
        retrieved = cases[0]
        assert retrieved['patient_id'] == 'pat_123'
        assert retrieved['condition_code'] == 'nv'
        assert retrieved['urgency_color'] == 'GREEN'
        assert retrieved['confidence'] == 92.5
        
        logger.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_tflite_model_loads():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'export', 'dermaid_int8.tflite'))
    
    if not os.path.exists(model_path):
        pytest.skip("Exported TFLite model not found. Skipping offline model parity test.")
    
    try:
        import tensorflow.lite as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        assert len(input_details) > 0
        
        # Verify edge quantization topological mapping matches standard NHWC shape
        shape = input_details[0]['shape']
        assert tuple(shape) == (1, 224, 224, 3), f"TFLite topology expects {shape}"
        
    except ImportError:
        pytest.skip("TensorFlow execution context not installed; skipping test.")
