import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from pathlib import Path

import config

def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                images, condition_labels, severity_labels, patient_ids = batch
            else:
                # Fallback if structure is different
                images, condition_labels = batch[0], batch[1]
                
            images = images.to(device)
            
            # Pass through backbone
            features = model.backbone(images)
            
            # Apply avgpool and flatten
            if hasattr(model, 'avgpool'):
                features = model.avgpool(features)
            elif hasattr(model.backbone, 'avgpool'):
                features = model.backbone.avgpool(features)
            elif features.dim() == 4:
                features = F.adaptive_avg_pool2d(features, (1, 1))
                
            features = torch.flatten(features, 1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(condition_labels.numpy())
            
    features_np = np.vstack(all_features)
    labels_np = np.concatenate(all_labels)
    
    return features_np, labels_np

def apply_feature_smote(features, labels, strategy='minority', k=5, seed=42):
    print("Class distribution before SMOTE:", Counter(labels))
    
    smote = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
    X_resampled, y_resampled = smote.fit_resample(features, labels)
    
    print("Class distribution after SMOTE:", Counter(y_resampled))
    
    return X_resampled, y_resampled

def compute_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)

class SMOTEAugmentedDataset(Dataset):
    def __init__(self, original_features, smote_features, smote_labels, transform=None):
        self.features = smote_features
        self.labels = smote_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feat = self.transform(feat)
            
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def run_smote_pipeline(model, train_loader, device):
    print("Extracting features from backbone...")
    features_np, labels_np = extract_features(model, train_loader, device)
    
    print("Applying SMOTE...")
    # 'not majority' could be a good choice, but 'minority' was default in instructions
    # We will pass strategy='auto' or similar if 'minority' fails on multi-class
    try:
        X_resampled, y_resampled = apply_feature_smote(features_np, labels_np, strategy='not majority', seed=config.RANDOM_SEED)
    except Exception as e:
        print(f"Warning: SMOTE strategy 'not majority' failed ({e}). Falling back to 'auto'.")
        X_resampled, y_resampled = apply_feature_smote(features_np, labels_np, strategy='auto', seed=config.RANDOM_SEED)
    
    print("Computing class weights...")
    class_weights_tensor = compute_class_weights(labels_np) # Compute weights on ORIGINAL distribution or new? Usually original.
    
    # Save processed features
    save_dir = config.DATA_DIR / 'processed'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'smote_features.npz'
    np.savez(save_path, features=X_resampled, labels=y_resampled)
    print(f"Resampled features saved to {save_path}")
    
    return X_resampled, y_resampled, class_weights_tensor
