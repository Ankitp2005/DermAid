import os
import sys
import argparse
import random
import time
import json
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import recall_score

# Add src to python path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
from model import DermAidModel
from dataset import get_dataloaders
from smote_pipeline import run_smote_pipeline
import train as train_module

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_recalls(model, val_loader, device):
    """Computes specific class recalls needed for tracking critical endpoints."""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in val_loader:
            images, cond_true = batch[0].to(device), batch[1].to(device)
            cond_logits, _, _ = model(images)
            probs = torch.softmax(cond_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_true.extend(cond_true.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    recalls = recall_score(y_true, y_pred, average=None, labels=range(7), zero_division=0)
    
    try:
        mel_idx = config.CLASS_NAMES.index('mel')
        bcc_idx = config.CLASS_NAMES.index('bcc')
        mel_recall = recalls[mel_idx]
        bcc_recall = recalls[bcc_idx]
    except ValueError:
        mel_recall, bcc_recall = 0.0, 0.0
        
    return mel_recall, bcc_recall

# Monkey-patch validate to log to wandb smoothly without altering original script
original_validate = train_module.validate

def wandb_patched_validate(model, val_loader, device):
    val_auc = original_validate(model, val_loader, device)
    mel_recall, bcc_recall = compute_recalls(model, val_loader, device)
    
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            'val_auc': val_auc,
            'mel_recall': mel_recall,
            'bcc_recall': bcc_recall
        })
        
    return val_auc

train_module.validate = wandb_patched_validate

# Patch print to intercept train loss and epoch
import builtins
original_print = builtins.print

def custom_print(*args, **kwargs):
    text = " ".join(str(a) for a in args)
    original_print(*args, **kwargs)
    
    if "Stage" in text and "Epoch" in text and "Train Loss" in text:
        try:
            parts = text.split("|")
            epoch_str = parts[0].split("Epoch")[1].split("/")[0].strip()
            train_loss = float(parts[1].split(":")[1].strip())
            
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({'epoch': int(epoch_str), 'train_loss': train_loss}, commit=False)
        except Exception:
            pass

builtins.print = custom_print

def main():
    parser = argparse.ArgumentParser(description="DermAid Training Orchestrator")
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to data directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--stage', type=str, default='both', choices=['1', '2', 'both'], help='Training stage')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Setup
    set_seed(config.RANDOM_SEED)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Initialized with device: {device}")

    if WANDB_AVAILABLE:
        wandb.init(project='dermaid-poc2026', config=vars(args))
    
    model = DermAidModel().to(device)
    
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Successfully resumed from checkpoint: {args.resume}")

    # Load Data
    print("Initializing Data Loaders...")
    loaders = get_dataloaders(args.data_dir, batch_size=config.BATCH_SIZE)
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    # Pre-compute SMOTE features and weights
    _, _, class_weights = run_smote_pipeline(model, train_loader, device)

    start_time = time.time()
    
    try:
        if args.stage in ['1', 'both']:
            model = train_module.train_stage1(model, train_loader, val_loader, device, class_weights, epochs=10)
            
        if args.stage in ['2', 'both']:
            model = train_module.train_stage2(model, train_loader, val_loader, device, class_weights, epochs=40)
            
    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt caught. Saving safety checkpoint...")
        base_dir = config.CHECKPOINT_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), base_dir / 'interrupted.pth')
        print(f"[!] Saved interrupted model to {base_dir / 'interrupted.pth'}")
        sys.exit(0)
        
    total_time = time.time() - start_time
    
    # Final Metrics for Table Output
    print("\nEvaluating final model for summary...")
    final_auc = original_validate(model, val_loader, device)
    mel_recall, bcc_recall = compute_recalls(model, val_loader, device)
    
    summary_data = {
        'best_val_auc': final_auc,
        'mel_recall': mel_recall,
        'bcc_recall': bcc_recall,
        'total_training_time': total_time
    }
    
    # Save log
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'logs/training_{timestamp}.json'
    with open(log_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    # Print Table
    print("\n" + "="*50)
    print("                TRAINING SUMMARY                  ")
    print("="*50)
    print(f" Best Val Macro AUC            : {final_auc:.4f}")
    print(f" Melanoma (mel) Recall         : {mel_recall:.4f}")
    print(f" Basal Cell Carcinoma Recall   : {bcc_recall:.4f}")
    print(f" Total Training Time           : {total_time/60:.2f} mins")
    print("="*50)
    print(f" JSON Summary saved to : {log_path}")
    print("="*50)

if __name__ == "__main__":
    main()
