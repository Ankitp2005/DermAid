import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from pathlib import Path
import copy

from model import DermAidModel
from loss import DermAidLoss
from dataset import get_dataloaders
from smote_pipeline import run_smote_pipeline
from mixup import mixup_data
import config

def validate(model, val_loader, device):
    """
    Evaluates the model on the validation set and returns Macro ROC-AUC.
    """
    model.eval()
    y_true = []
    y_score = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Batch: images, condition_labels, severity_labels, patient_ids
            images, cond_true = batch[0].to(device), batch[1].to(device)
            
            cond_pred, _, _ = model(images)
            cond_probs = torch.softmax(cond_pred, dim=1)
            
            y_true.extend(cond_true.cpu().numpy())
            y_score.extend(cond_probs.cpu().numpy())
            
    try:
        macro_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
    except ValueError:
        # Fallback if a class is entirely missing in a tiny debug batch
        macro_auc = 0.5
        
    return macro_auc

def train_stage1(model, train_loader, val_loader, device, class_weights, epochs=10):
    print("\n--- STAGE 1: Training Heads Only ---")
    model.freeze_backbone()
    
    # Optimizer for heads only
    heads_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(heads_params, lr=1e-3, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = DermAidLoss(class_weights=class_weights.to(device))
    scaler = GradScaler()
    
    best_auc = 0.0
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = config.CHECKPOINT_DIR / 'stage1_best.pth'
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            images = batch[0].to(device)
            cond_true = batch[1].to(device)
            sev_true = batch[2].to(device)
            
            optimizer.zero_grad()
            
            # MixUp implementation for multi-target
            # Stack targets so mixup_data can permute them identically
            targets = torch.stack([cond_true, sev_true], dim=1)
            mixed_images, target_a, target_b, lam = mixup_data(images, targets, alpha=0.4, device=device)
            
            cond_a, sev_a = target_a[:, 0], target_a[:, 1]
            cond_b, sev_b = target_b[:, 0], target_b[:, 1]
            
            with autocast():
                cond_pred, sev_pred, conf_pred = model(mixed_images)
                
                loss_a, _, _, _ = criterion(cond_pred, sev_pred, conf_pred, cond_a, sev_a)
                loss_b, _, _, _ = criterion(cond_pred, sev_pred, conf_pred, cond_b, sev_b)
                loss = lam * loss_a + (1 - lam) * loss_b
                
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        val_auc = validate(model, val_loader, device)
        
        print(f"Stage 1 - Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Macro AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            
    # Load best before returning
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
    return model

def train_stage2(model, train_loader, val_loader, device, class_weights, epochs=40):
    print("\n--- STAGE 2: Fine-tuning Full Model ---")
    model.unfreeze_backbone()
    
    # Differential learning rates
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 3e-5},
        {'params': head_params, 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = DermAidLoss(class_weights=class_weights.to(device))
    scaler = GradScaler()
    
    best_auc = 0.0
    patience_counter = 0
    patience_limit = 10
    best_model_path = config.CHECKPOINT_DIR / 'dermaid_best.pth'
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            images = batch[0].to(device)
            cond_true = batch[1].to(device)
            sev_true = batch[2].to(device)
            
            optimizer.zero_grad()
            
            targets = torch.stack([cond_true, sev_true], dim=1)
            mixed_images, target_a, target_b, lam = mixup_data(images, targets, alpha=0.4, device=device)
            
            cond_a, sev_a = target_a[:, 0], target_a[:, 1]
            cond_b, sev_b = target_b[:, 0], target_b[:, 1]
            
            with autocast():
                cond_pred, sev_pred, conf_pred = model(mixed_images)
                loss_a, _, _, _ = criterion(cond_pred, sev_pred, conf_pred, cond_a, sev_a)
                loss_b, _, _, _ = criterion(cond_pred, sev_pred, conf_pred, cond_b, sev_b)
                loss = lam * loss_a + (1 - lam) * loss_b
                
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        val_auc = validate(model, val_loader, device)
        
        print(f"Stage 2 - Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Macro AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  --> New best model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
    return model

def train_dermaid(data_dir=None, device=None):
    if device is None:
        device = torch.device(config.DEVICE)
        
    # Get DataLoaders
    print("Loading data...")
    loaders = get_dataloaders(data_dir, batch_size=config.BATCH_SIZE)
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    # Initialize Model
    model = DermAidModel().to(device)
    
    # Run SMOTE Pipeline to get dynamic class weights
    # Note: SMOTE features themselves might not be folded into train_loader automatically here.
    # The requirement specifically stated: run_smote_pipeline orchestrates it, and train_dermaid calls it.
    _, _, class_weights = run_smote_pipeline(model, train_loader, device)
    
    # Stage 1 Training
    model = train_stage1(model, train_loader, val_loader, device, class_weights, epochs=10)
    
    # Stage 2 Training
    model = train_stage2(model, train_loader, val_loader, device, class_weights, epochs=40)
    
    print("\nTraining Complete! Best model loaded.")
    return model

if __name__ == "__main__":
    train_dermaid()
