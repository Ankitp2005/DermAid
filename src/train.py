import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Import local modules
from src.dataset import HAM10000Dataset
from src.augmentation import get_train_transforms, get_val_transforms
from src.model import DermAidModel
from src.loss import DermAidLoss


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    MixUp: blend two training samples to smooth decision boundaries.
    Especially effective for nv vs mel (visually similar but clinically opposite).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def validate(model, val_loader, device):
    """Validation function that returns macro-averaged AUC"""
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, cond_labels, sev_labels in val_loader:
            images = images.to(device)
            cond_pred, _, _ = model(images)
            probs = torch.softmax(cond_pred, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(cond_labels.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)

    # MACRO-AVERAGED AUC — as promised in PPT
    # One-vs-rest for multiclass
    macro_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    return macro_auc


def train_dermaid(
    model, train_loader, val_loader, num_epochs=50, device="cuda", class_weights=None
):
    model = model.to(device)
    criterion = DermAidLoss(class_weights=class_weights.to(device))

    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999)
    )

    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Mixed precision training for speed
    scaler = GradScaler()

    # Early stopping
    best_val_auc = 0.0
    patience_counter = 0
    patience = 10

    for epoch in range(num_epochs):
        # ── TRAINING ─────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch_idx, (images, cond_labels, sev_labels) in enumerate(train_loader):
            images = images.to(device)
            cond_labels = cond_labels.to(device)
            sev_labels = sev_labels.to(device)

            # MixUp augmentation (applied during training)
            images, targets_a, targets_b, lam = mixup_data(
                images, cond_labels, alpha=0.4, device=device
            )

            optimizer.zero_grad()

            with autocast():  # mixed precision
                cond_pred, sev_pred, conf_pred = model(images)
                loss, l_cond, l_sev, l_conf = criterion(
                    cond_pred, sev_pred, conf_pred, cond_labels, sev_labels
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # ── VALIDATION ───────────────────────────────────────────────────
        val_auc = validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Macro-AUC: {val_auc:.4f}"
        )

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "dermaid_best.pth")
            patience_counter = 0
            print(f"  ✓ New best model saved! AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    return model


if __name__ == "__main__":
    # This is just to show how the training would be called
    # Actual training would be done in a separate script or notebook
    print("Training script ready. Use this in your training notebook or script.")
