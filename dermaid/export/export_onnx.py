import os
import sys
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

# Add src to Python's module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import DermAidModel

def export_to_onnx(checkpoint_path, output_path='export/dermaid.onnx'):
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Initialize the model 
    model = DermAidModel()
    
    # Handle missing CUDA cleanly if running on CPU-only machines
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    print(f"Exporting model to {output_path}...")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['condition_logits', 'severity_logits', 'confidence'],
        dynamic_axes={'input_image': {0: 'batch_size'}}
    )
    
    print("ONNX export complete. Verifying model structure...")
    
    # Verify the structure with the checker
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model checker passed!")
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported Model Size: {size_mb:.2f} MB")
    
    return output_path


def verify_onnx_inference(onnx_path, test_image_path=None):
    print("\nVerifying ONNX runtime inference...")
    
    try:
        ort_session = ort.InferenceSession(onnx_path)
        
        dummy_input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        outputs = ort_session.run(
            ['condition_logits', 'severity_logits', 'confidence'], 
            {'input_image': dummy_input_np}
        )
        
        cond_out, sev_out, conf_out = outputs
        
        expected_cond = (1, 7)
        expected_sev = (1, 3)
        expected_conf = (1, 1)
        
        shapes_correct = (cond_out.shape == expected_cond) and \
                         (sev_out.shape == expected_sev) and \
                         (conf_out.shape == expected_conf)
                         
        print(f"Condition Logits Shape : {cond_out.shape} -> Expected: {expected_cond}")
        print(f"Severity Logits Shape  : {sev_out.shape} -> Expected: {expected_sev}")
        print(f"Confidence Shape       : {conf_out.shape} -> Expected: {expected_conf}")
        
        if shapes_correct:
            print("[STATUS]: PASS")
        else:
            print("[STATUS]: FAIL - Shape mismatch in ONNX outputs.")
            
    except Exception as e:
        print(f"[STATUS]: FAIL - Inference threw an exception: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export PyTorch DermAid Model to ONNX")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='export/dermaid.onnx', help='Path to save ONNX model')
    args = parser.parse_args()
    
    saved_onnx_path = export_to_onnx(args.checkpoint, args.output)
    verify_onnx_inference(saved_onnx_path)
