import os
import sys
import time
import json
import sqlite3
import numpy as np

# Ensure Python path sees src and api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'api')))

# Terminal colors for output styling
CGREEN = '\033[92m'
CRED = '\033[91m'
CRESET = '\033[0m'

results = []
TOTAL_CHECKS = 15

def pass_check(name):
    results.append(True)
    print(f"{CGREEN}✅ PASS: {name}{CRESET}")

def fail_check(name, reason):
    results.append(False)
    print(f"{CRED}❌ FAIL: {name} — {reason}{CRESET}")

def run_validation():
    print("Starting DermAid Submission Validation...\n" + "="*55)
    
    # 1. TFLite exists & constraints
    tflite_path = "export/dermaid_int8.tflite"
    if os.path.exists(tflite_path):
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        if size_mb <= 14.0:
            pass_check(f"dermaid_int8.tflite exists and size <= 14 MB ({size_mb:.1f} MB)")
        else:
            fail_check("dermaid_int8.tflite size limit", f"Size is {size_mb:.1f} MB, threshold is 14.0 MB")
    else:
        fail_check("dermaid_int8.tflite exists", f"File '{tflite_path}' missing. Note: This assumes export ran successfully.")
        
    # 2. TFLite model loads and shape matches
    interpreter = None
    try:
        import tensorflow.lite as tflite
        if os.path.exists(tflite_path):
            interpreter = tflite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            shape = interpreter.get_input_details()[0]['shape']
            if tuple(shape) == (1, 224, 224, 3):
                pass_check("TFLite model loads and input shape is (1,224,224,3)")
            else:
                fail_check("TFLite input shape", f"Got irregular dimensions: {tuple(shape)}")
        else:
            fail_check("TFLite model loads and input shape", "Model missing")
    except Exception as e:
        fail_check("TFLite model loads validation", str(e))
        
    # 3. TFLite CPU inference ceiling
    try:
        if interpreter:
            # Generate random synthetic data that aligns with expected type
            dtype = interpreter.get_input_details()[0]['dtype']
            if dtype == np.float32:
                dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            else:
                dummy_input = np.random.randint(0, 255, (1, 224, 224, 3), dtype=dtype)
                
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], dummy_input)
            
            # Flush pipeline warmup overhead
            interpreter.invoke()
            
            start = time.time()
            interpreter.invoke()
            ms = (time.time() - start) * 1000
            
            if ms < 3000:
                pass_check(f"TFLite inference completes in < 3000ms on CPU ({ms:.1f} ms)")
            else:
                fail_check("TFLite inference time barrier Exceeded", f"Clocked at {ms:.1f} ms")
        else:
            fail_check("TFLite inference completes in < 3000ms", "Engine isolated / Model absent")
    except Exception as e:
        fail_check("TFLite inference completes", str(e))
        
    # 4 to 6. Evaluation metrics enforcement
    eval_path = "results/evaluation_results.json"
    if os.path.exists(eval_path):
        try:
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
                
            auc = eval_data.get('macro_auc', 0.0)
            if auc >= 0.91:
                pass_check(f"macro_auc >= 0.91 (Achieved: {auc:.3f})")
            else:
                fail_check("macro_auc >= 0.91 target", f"Deficient value: {auc:.3f}")
                
            mel = eval_data.get('mel_recall', 0.0)
            if mel >= 0.90:
                pass_check(f"mel_recall >= 0.90 (Achieved: {mel:.3f})")
            else:
                fail_check("mel_recall >= 0.90 target", f"Deficient value: {mel:.3f}")
                
            bcc = eval_data.get('bcc_recall', 0.0)
            if bcc >= 0.85:
                pass_check(f"bcc_recall >= 0.85 (Achieved: {bcc:.3f})")
            else:
                fail_check("bcc_recall >= 0.85 target", f"Deficient value: {bcc:.3f}")
                
        except Exception as e:
            fail_check("Evaluation parsing integrity", str(e))
            fail_check("mel_recall checks", "Fatal json exception")
            fail_check("bcc_recall checks", "Fatal json exception")
    else:
        # Graceful failure explanations to assist users in resolving gaps
        fail_check("results/evaluation_results.json exists and macro_auc >= 0.91", "JSON missing. Run test script.")
        fail_check("mel_recall >= 0.90 from evaluation results", "JSON missing.")
        fail_check("bcc_recall >= 0.85 from evaluation results", "JSON missing.")

    # 7 to 9. Referential Integrity validation across classes and translations
    try:
        from config import CLASS_NAMES, SEVERITY_MAP
        from referral_engine import REFERRAL_MATRIX_EN, REFERRAL_MATRIX_HI
        
        # English Matrix Verification
        matched_en = 0
        for c in CLASS_NAMES:
            if (c, SEVERITY_MAP.get(c, 'None')) in REFERRAL_MATRIX_EN: matched_en += 1
            
        if matched_en == 7:
            pass_check("All 7 condition classes present in REFERRAL_MATRIX")
        else:
            fail_check("REFERRAL_MATRIX mapping", f"Only {matched_en}/7 matched structurally.")
            
        # Clinical Spectrum Extent Verification 
        tiers = set(SEVERITY_MAP.values())
        if len(tiers) == 3 and 'Low Risk' in tiers and 'Refer Immediately' in tiers:
            pass_check("All 3 severity tiers present in outputs")
        else:
            fail_check("Severity tiers integrity", f"Discovered aberrant set: {tiers}")
            
        # Hindi Matrix Localization Verification
        matched_hi = 0
        for c in CLASS_NAMES:
            if (c, SEVERITY_MAP.get(c, 'None')) in REFERRAL_MATRIX_HI: matched_hi += 1
            
        if matched_hi == 7:
            pass_check("Hindi translations present for all 7 conditions")
        else:
            fail_check("Hindi Localization Completeness", f"Only mapped {matched_hi}/7 conditions")
            
    except Exception as e:
        fail_check("REFERRAL_MATRIX condition verification", f"Exception evaluating dictionary mappings: {e}")
        fail_check("Severity tiers enumeration", "Skipped due to crash")
        fail_check("Hindi translations mapping", "Skipped due to crash")

    # 10. Hardware-agnostic computer vision check structure validation
    try:
        from image_quality import check_image_quality
        res = check_image_quality(np.zeros((100, 100, 3), dtype=np.uint8))
        required_keys = ['is_blurry', 'too_dark', 'too_bright', 'low_contrast', 'is_usable', 'quality_score', 'message']
        if all(k in res for k in required_keys):
            pass_check("check_image_quality returns proper dict structure")
        else:
            fail_check("check_image_quality payload signature", f"Schema malformed, missing required elements.")
    except Exception as e:
        fail_check("check_image_quality implementation structure test", str(e))
        
    # 11. SQLite functionality check mapping
    try:
        import tempfile
        from case_logger import CaseLogger
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            path = tf.name
            
        logger = CaseLogger(path)
        dummy_row = {'condition_code': 'sys_test', 'severity': 'Low Risk', 'urgency_color': 'GREEN'}
        idx = logger.log_case('pat_1', 'worker_A', 'phc', 'sys/file', dummy_row)
        if idx > 0:
            pass_check("SQLite case_logger creates table and inserts successfully")
        else:
            fail_check("SQLite insertion pipeline", "Insert rowid failed.")
        logger.close()
        os.remove(path)
    except Exception as e:
        fail_check("SQLite configuration execution", str(e))

    # 12. Structural dependency trace: GradCAM
    try:
        import gradcam
        pass_check("GradCAM imports without error")
    except Exception as e:
        fail_check("GradCAM encapsulation syntax", str(e))
        
    # 13. Structural dependency trace: FastAPI Engine
    try:
        from main import app
        pass_check("FastAPI app imports without error")
    except Exception as e:
        fail_check("FastAPI ASGI compatibility", str(e))


    # 14. Documentation completeness Check
    try:
        if os.path.exists("README.md") and os.path.getsize("README.md") > 500:
            pass_check("README.md exists and is > 500 bytes")
        else:
            fail_check("README.md audit", "File misses sizing heuristics or was not generated.")
    except Exception as e:
         fail_check("README.md evaluation", str(e))
         
    # 15. Demonstrator payload location verify
    try:
        if os.path.exists("notebooks/04_Demo.ipynb"):
            pass_check("notebooks/04_Demo.ipynb exists")
        else:
            fail_check("notebooks/04_Demo.ipynb file linkage", "Demo configuration notebook absent")
    except Exception as e:
         fail_check("notebooks/04_Demo.ipynb linkage verification", str(e))
         
    # Conclusion execution
    passed = sum(results)
    
    print("\n" + "="*55)
    
    # Ternary switch formatting
    status_str = "READY TO SUBMIT" if passed == TOTAL_CHECKS else "NOT READY"
    final_color = CGREEN if passed == TOTAL_CHECKS else CRED
    
    print(f"{final_color}{passed}/{TOTAL_CHECKS} checks passed. [{status_str}]{CRESET}")
    
if __name__ == '__main__':
    run_validation()
