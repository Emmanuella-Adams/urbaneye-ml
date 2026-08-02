import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ------------------------------
# PyTorch Dataset Definition
# ------------------------------
class BuildingFootprintDataset(Dataset):
    """
    PyTorch Dataset for Satellite Image Tile & Building Footprint Mask Pairs.
    """
    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.image_dir = os.path.abspath(image_dir)
        self.mask_dir = os.path.abspath(mask_dir)
        self.transform = transform
        self.augment = augment

        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.png')) + 
                                  glob.glob(os.path.join(self.image_dir, '*.jpg')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')) + 
                                 glob.glob(os.path.join(self.mask_dir, '*.jpg')))

        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch in image ({len(self.image_paths)}) and mask ({len(self.mask_paths)}) file counts."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load RGB image and grayscale binary mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Convert to numpy float arrays
        image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize RGB to [0, 1]
        mask_np = np.array(mask, dtype=np.float32) / 255.0   # Binary 0.0 or 1.0

        # Threshold mask to strictly 0 or 1
        mask_np = (mask_np > 0.5).astype(np.float32)

        # Simple spatial augmentations (Random Flips & Rotations for training)
        if self.augment:
            if np.random.rand() > 0.5:
                image_np = np.fliplr(image_np).copy()
                mask_np = np.fliplr(mask_np).copy()
            if np.random.rand() > 0.5:
                image_np = np.flipud(image_np).copy()
                mask_np = np.flipud(mask_np).copy()

        # Transpose image to (C, H, W) for PyTorch
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # Shape: (1, H, W)

        return image_tensor, mask_tensor

# ------------------------------
# DataLoader Creation Factory
# ------------------------------
def create_dataloaders(image_dir='sample_data/images', mask_dir='sample_data/masks', 
                       batch_size=4, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    """
    Split dataset into Train (70%), Val (15%), and Test (15%), returning PyTorch DataLoaders.
    """
    full_dataset = BuildingFootprintDataset(image_dir, mask_dir, augment=False)
    total_len = len(full_dataset)

    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)
    test_size = total_len - train_size - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Enable augmentation for training subset
    train_dataset.dataset.augment = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
