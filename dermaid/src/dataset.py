import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from torchvision import transforms

import config

class HAM10000Dataset(Dataset):
    def __init__(self, metadata_csv, img_dir, transform=None, mode='train'):
        if isinstance(metadata_csv, (str, Path)):
            self.metadata = pd.read_csv(metadata_csv)
        else:
            self.metadata = metadata_csv.reset_index(drop=True)
            
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        
        self.dx_to_idx = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}
        
        sev_str_to_int = {
            'Low Risk': 0,
            'Refer Soon': 1,
            'Refer Immediately': 2
        }
        self.severity_map = {dx: sev_str_to_int[sev] for dx, sev in config.SEVERITY_MAP.items()}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row['image_id']
        dx = row['dx']
        patient_id = row['patient_id']
        
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                try:
                    # Try passing as NumPy array for Albumentations
                    augmented = self.transform(image=np.array(image))
                    image = augmented['image']
                except TypeError:
                    # Fallback to Torchvision
                    image = self.transform(image)
        except Exception as e:
            # Handle missing images gracefully
            image = torch.zeros((3, config.IMG_SIZE, config.IMG_SIZE))
            
        condition_label = self.dx_to_idx[dx]
        severity_label = self.severity_map[dx]
        
        return image, condition_label, severity_label, patient_id

def patient_level_split(metadata_df, train_ratio=0.70, val_ratio=0.15, seed=42):
    # Split by patient_id, stratified by dx
    patient_df = metadata_df.groupby('patient_id').first().reset_index()
    
    test_ratio = 1.0 - train_ratio - val_ratio
    
    train_patients, temp_patients = train_test_split(
        patient_df, 
        test_size=(val_ratio + test_ratio), 
        stratify=patient_df['dx'], 
        random_state=seed
    )
    
    val_patients, test_patients = train_test_split(
        temp_patients, 
        test_size=(test_ratio / (val_ratio + test_ratio)), 
        stratify=temp_patients['dx'], 
        random_state=seed
    )
    
    train_df = metadata_df[metadata_df['patient_id'].isin(train_patients['patient_id'])].reset_index(drop=True)
    val_df = metadata_df[metadata_df['patient_id'].isin(val_patients['patient_id'])].reset_index(drop=True)
    test_df = metadata_df[metadata_df['patient_id'].isin(test_patients['patient_id'])].reset_index(drop=True)
    
    return train_df, val_df, test_df

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    data_dir = Path(data_dir)
    metadata_path = data_dir / "HAM10000_metadata.csv"
    img_dir = data_dir / "images" # Modify if images are saved in "raw" subfolder
    
    if metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path)
    else:
        metadata_df = pd.DataFrame(columns=['image_id', 'dx', 'dx_type', 'age', 'sex', 'localization', 'patient_id'])
    
    if len(metadata_df) > 0:
        train_df, val_df, test_df = patient_level_split(
            metadata_df, 
            train_ratio=0.70, 
            val_ratio=0.15, 
            seed=config.RANDOM_SEED
        )
    else:
        train_df, val_df, test_df = metadata_df, metadata_df, metadata_df

    train_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    train_dataset = HAM10000Dataset(train_df, img_dir, transform=train_transforms, mode='train')
    val_dataset = HAM10000Dataset(val_df, img_dir, transform=val_transforms, mode='val')
    test_dataset = HAM10000Dataset(test_df, img_dir, transform=val_transforms, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
