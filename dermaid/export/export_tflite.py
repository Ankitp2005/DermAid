import os
import sys
import numpy as np
import tensorflow as tf
import subprocess

def convert_onnx_to_tflite(onnx_path, tflite_path="export/dermaid_int8.tflite"):
    print(f"Step 1: Converting {onnx_path} to TensorFlow SavedModel using onnx2tf...")
    saved_model_dir = "export/saved_model"
    
    # Run onnx2tf to convert ONNX to TF SavedModel
    # -osd forces it to physically dump the SavedModel directory so we can run our custom INT8 pass on it.
    try:
        subprocess.run(["onnx2tf", "-i", onnx_path, "-o", saved_model_dir, "-osd"], check=True)
    except FileNotFoundError:
        print("Error: onnx2tf not found. Please install it via 'pip install onnx2tf'")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error during onnx2tf conversion: {e}")
        sys.exit(1)
        
    print("\nStep 2: Converting SavedModel to TFLite (INT8 Quantization) ...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Enable quantization optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset generator for INT8 calibration
    # Generates standard normal normalized inputs (simulating normalized ImageNet batches)
    def representative_dataset():
        for _ in range(100):
            # Input shape should match whatever ONNX exported.
            # PyTorch MobileNetV3 expects NCHW (1, 3, 224, 224)
            data = np.random.randn(1, 3, 224, 224).astype(np.float32)
            yield [data]
            
    converter.representative_dataset = representative_dataset
    
    # Target pure INT8 operations for Edge safety & size reduction
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # We keep the outer input/output tensors as Float32 so Android Kotlin code
    # doesn't need to manually handle INT8 dequantization formulas.
    converter.inference_input_type = tf.float32 
    converter.inference_output_type = tf.float32

    print("Quantizing weights... This may take a minute...")
    tflite_quant_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)
        
    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print("="*50)
    print(f"✅ TFLite INT8 Export Complete!")
    print(f"✅ Output Location: {tflite_path}")
    print(f"✅ Final Model Size: {size_mb:.2f} MB (Target < 14MB)")
    print("="*50)
    print("You can now download this file and place it in the Android assets folder!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_tflite.py <path_to_onnx>")
        sys.exit(1)
        
    onnx_file = sys.argv[1]
    
    # Ensure export directory exists
    os.makedirs("export", exist_ok=True)
    
    convert_onnx_to_tflite(onnx_file)
