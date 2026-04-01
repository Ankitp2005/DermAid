import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import DermAidModel

def size_mb(path): return os.path.getsize(path) / 1024**2 if os.path.exists(path) else 0.0

tf_path = os.path.join(os.path.dirname(__file__), 'dermaid_int8.tflite')
pt_path = os.path.join(os.path.dirname(__file__), '../checkpoints/dermaid_best.pth')

t_mb, p_mb = size_mb(tf_path), size_mb(pt_path)
model = DermAidModel()

calc = lambda mod: sum(p.numel() for p in mod.parameters() if p.requires_grad)
b_params = calc(model.backbone)
h_params = calc(model.condition_head) + calc(model.severity_head) + calc(model.confidence_head)

input_ram_mb = (1 * 3 * 224 * 224 * 4) / 1024**2

print(f"TFLite Model Size: {t_mb:.2f} MB")
print(f"PyTorch Checkpoint Size: {p_mb:.2f} MB")
print(f"Parameters Breakdown: {b_params:,} (Backbone) | {h_params:,} (Heads) | {b_params+h_params:,} (Total)")
print(f"TFLite Target <= 14 MB: {'PASS' if 0 < t_mb <= 14 else 'FAIL'}")
print(f"Estimated Minimum Inference RAM: {input_ram_mb + t_mb:.2f} MB (Model + Input Tensor)")
