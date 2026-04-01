import os
import sys
import time
import argparse
import tracemalloc
import numpy as np
import torch

# Inject contextual root path for core model loading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import DermAidModel

def benchmark_tflite(interpreter, dummy_data, n_runs=100):
    tracemalloc.start()
    times = []
    
    input_index = interpreter.get_input_details()[0]['index']
    
    # Pre-execute Warmup to discard engine initialization overhead
    interpreter.set_tensor(input_index, dummy_data)
    interpreter.invoke()
    
    tracemalloc.reset_peak()
    
    for _ in range(n_runs):
        interpreter.set_tensor(input_index, dummy_data)
        
        start = time.perf_counter()
        interpreter.invoke()
        ms = (time.perf_counter() - start) * 1000.0
        
        times.append(ms)
        
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    latency = np.array(times)
    throughput = 1000.0 / np.mean(latency)
    return latency, throughput, peak / (1024*1024)

def benchmark_pytorch(model, dummy_data, device_name, n_runs=100):
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    data = dummy_data.to(device)
    
    times = []
    
    # Engine Warmup Pass
    with torch.no_grad():
        model(data)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(data)
            if device.type == 'cuda':
                # Critical for preventing async execution artifacts in measurement
                torch.cuda.synchronize()
                
        ms = (time.perf_counter() - start) * 1000.0
        times.append(ms)
        
    latency = np.array(times)
    throughput = 1000.0 / np.mean(latency)
    return latency, throughput

def benchmark_all(tflite_path, pytorch_checkpoint, test_loader=None, device='cpu'):
    print("\nStarting DermAid Edge-Hardware Benchmarks...\n")
    
    n_runs = 100
    
    # Secure appropriate payload tensors
    if test_loader is not None:
        batch = next(iter(test_loader))
        img_pt = batch[0][0:1] # Ensure isolated batch size parameter (1, 3, 224, 224)
        if img_pt.shape[0] == 0:
            img_pt = torch.randn(1, 3, 224, 224)
    else:
        img_pt = torch.randn(1, 3, 224, 224)
    
    # Bootstrapping structural PyTorch validation context 
    pt_model = DermAidModel()
    if os.path.exists(pytorch_checkpoint):
        try:
            pt_model.load_state_dict(torch.load(pytorch_checkpoint, map_location='cpu'))
        except Exception:
            print("Warning: Checkpoint load crashed, defaulting to base initialization.")
    else:
        print("Warning: Checkpoint not found, evaluating structural baseline speeds.")
        
    # Bootstrapping TFLite Edge validation context
    try:
        import tensorflow.lite as tflite
        interpreter = tflite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        dtype = interpreter.get_input_details()[0]['dtype']
        if dtype == np.float32:
            img_tf = img_pt.numpy()
        else: # Handle INT8 fallback payload simulations
            img_tf = np.random.randint(-128, 127, (1, 224, 224, 3), dtype=dtype)
            
        tf_latency, tf_tp, tf_peak_ram = benchmark_tflite(interpreter, img_tf, n_runs=n_runs)
        has_tflite = True
    except Exception as e:
        print(f"Skipping TFLite evaluation -> Context Initialization Failed: {e}")
        has_tflite = False
        
    # Native Hardware Executions
    pt_cpu_lat, pt_cpu_tp = benchmark_pytorch(pt_model, img_pt, 'cpu', n_runs)
    
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        pt_gpu_lat, pt_gpu_tp = benchmark_pytorch(pt_model, img_pt, 'cuda', n_runs)
        
    # Presentation Formalities
    print("="*85)
    print(f"{'Environment Unit':<20} | {'Avg (ms)':<9} | {'Min (ms)':<9} | {'Max (ms)':<9} | {'P95 (ms)':<9} | {'Throughput'}")
    print("-" * 85)
    
    def print_row(name, lat, tp):
        print(f"{name:<20} | {np.mean(lat):<9.2f} | {np.min(lat):<9.2f} | {np.max(lat):<9.2f} | {np.percentile(lat, 95):<9.2f} | {tp:.1f} i/s")
        
    if has_tflite:
        print_row("TFLite INT8 (CPU)", tf_latency, tf_tp)
        
    print_row("PyTorch (CPU)", pt_cpu_lat, pt_cpu_tp)
    
    if has_gpu:
        print_row("PyTorch (GPU)", pt_gpu_lat, pt_gpu_tp)
        
    print("=" * 85)
    
    # Contextual Verdict Printout targeting mobile environment specs
    if has_tflite:
        print(f"\n>> TFLite Peak Working Memory Footprint: {tf_peak_ram:.2f} MB")
        
        tf_avg = np.mean(tf_latency)
        verdict = "PASS" if tf_avg <= 3000 else "FAIL"
        
        print("\n" + "="*65)
        color = '\033[92m' if verdict == "PASS" else '\033[91m'
        print(f"{color}TFLite INT8 is {tf_avg:.1f}ms avg — TARGET: <3000ms — {verdict}\033[0m")
        print("=" * 65 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-platform latency & memory inference profiling")
    parser.add_argument('--tflite_path', type=str, default='dermaid_int8.tflite', 
                        help="Path referencing optimized TFLite binary")
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/dermaid_best.pth',
                        help="Path referencing structured PyTorch weights")
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    benchmark_all(
        tflite_path=args.tflite_path, 
        pytorch_checkpoint=args.checkpoint, 
        test_loader=None, # Inferred isolation dummy
        device=args.device
    )
