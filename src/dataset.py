import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class HAM10000Dataset(Dataset):
    """
    HAM10000 Dataset class for loading and preprocessing skin lesion images.
    Handles train/val/test splits and applies domain adaptation transforms.
    """

    def __init__(
        self,
        data_dir,
        metadata_path,
        transform=None,
        split="train",
        validation_split=0.15,
        test_split=0.15,
        random_state=42,
    ):
        """
        Args:
            data_dir (string): Directory with all the images.
            metadata_path (string): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            split (string): One of 'train', 'val', or 'test'.
            validation_split (float): Proportion of dataset to use for validation.
            test_split (float): Proportion of dataset to use for testing.
            random_state (int): Random seed for reproducibility.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split

        # Load metadata
        self.metadata = pd.read_csv(metadata_path)

        # Map diagnosis to integer labels
        self.class_to_idx = {
            "nv": 0,  # Melanocytic Nevi
            "mel": 1,  # Melanoma
            "bkl": 2,  # Benign Keratosis-like
            "bcc": 3,  # Basal Cell Carcinoma
            "akiec": 4,  # Actinic Keratosis / Bowen's
            "vasc": 5,  # Vascular Lesions
            "df": 6,  # Dermatofibroma
        }

        # Filter out any rows with missing image IDs or diagnoses
        self.metadata = self.metadata.dropna(subset=["image_id", "dx"])

        # Create label mapping
        self.metadata["label"] = self.metadata["dx"].map(self.class_to_idx)

        # Simple random split for now (we'll improve this later)
        np.random.seed(random_state)
        n_samples = len(self.metadata)
        indices = np.random.permutation(n_samples)

        train_end = int(n_samples * (1 - validation_split - test_split))
        val_end = int(n_samples * (1 - test_split))

        if split == "train":
            self.metadata = self.metadata.iloc[indices[:train_end]]
        elif split == "val":
            self.metadata = self.metadata.iloc[indices[train_end:val_end]]
        else:  # test
            self.metadata = self.metadata.iloc[indices[val_end:]]

        # Reset index
        self.metadata = self.metadata.reset_index(drop=True)

        # Print dataset info
        print(f"{split.upper()} dataset: {len(self.metadata)} samples")
        print("Class distribution:")
        print(self.metadata["dx"].value_counts())

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get image name and label
        img_name = os.path.join(
            self.data_dir, self.metadata.iloc[idx]["image_id"] + ".jpg"
        )
        label = self.metadata.iloc[idx]["label"]

        # Load image
        try:
            image = Image.open(img_name).convert("RGB")
        except FileNotFoundError:
            # Try with different extensions if .jpg fails
            for ext in [".png", ".jpeg", ".JPG", ".PNG"]:
                alt_img_name = os.path.join(
                    self.data_dir, self.metadata.iloc[idx]["image_id"] + ext
                )
                if os.path.exists(alt_img_name):
                    image = Image.open(alt_img_name).convert("RGB")
                    break
            else:
                raise FileNotFoundError(
                    f"Image not found for ID: {self.metadata.iloc[idx]['image_id']}"
                )

        # Apply transforms
        if self.transform:
            # Convert PIL image to numpy array for albumentations
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented["image"]
            # Convert to torch tensor and permute dimensions if needed
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, label
