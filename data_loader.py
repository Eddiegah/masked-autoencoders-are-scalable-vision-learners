"""
Data loading utilities for MAE training.
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image


class ImageNet1KDataLoader:
    """DataLoader for ImageNet-1K dataset."""
    
    def __init__(
        self,
        data_path,
        batch_size,
        num_workers=4,
        pin_memory=True,
        img_size=224,
        split='train',
    ):
        """
        Args:
            data_path: Path to ImageNet-1K root directory
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
            img_size: Input image size (default 224)
            split: 'train' or 'val'
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.img_size = img_size
        self.split = split
        
        # NOTE: Standard ImageNet-1K preprocessing
        # Normalize with ImageNet statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def get_transforms(self):
        """Get data augmentation transforms."""
        if self.split == 'train':
            # NOTE: MAE uses relatively simple augmentation during pretraining
            # Standard crop, horizontal flip, and normalization
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=(0.2, 1.0),
                    interpolation=Image.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:  # val split
            transform = transforms.Compose([
                transforms.Resize(
                    int(self.img_size * 256 / 224),
                    interpolation=Image.BICUBIC
                ),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        
        return transform
    
    def get_dataloader(self):
        """Create and return DataLoader."""
        transform = self.get_transforms()
        
        dataset = datasets.ImageNet(
            root=self.data_path,
            split=self.split,
            transform=transform,
        )
        
        # NOTE: Use DistributedSampler for multi-GPU training
        # For single GPU, standard RandomSampler is used
        shuffle = (self.split == 'train')
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=(self.split == 'train'),
        )
        
        return dataloader


class CifarDataLoader:
    """Simple DataLoader for CIFAR-10 (for local testing without full ImageNet)."""
    
    def __init__(
        self,
        data_path,
        batch_size,
        num_workers=4,
        pin_memory=True,
        split='train',
    ):
        """
        Args:
            data_path: Path to CIFAR-10 root directory
            batch_size: Batch size
            num_workers: Number of workers
            pin_memory: Pin memory flag
            split: 'train' or 'test'
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.split = split
        
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2470, 0.2435, 0.2616]
    
    def get_transforms(self):
        """Get transforms for CIFAR-10."""
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        
        return transform
    
    def get_dataloader(self):
        """Create and return CIFAR-10 DataLoader."""
        transform = self.get_transforms()
        is_train = (self.split == 'train')
        
        dataset = datasets.CIFAR10(
            root=self.data_path,
            train=is_train,
            download=True,
            transform=transform,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=is_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=is_train,
        )
        
        return dataloader


def get_dataloader(dataset_name, data_path, batch_size, split='train', **kwargs):
    """
    Get dataloader by dataset name.
    
    Args:
        dataset_name: 'imagenet' or 'cifar10'
        data_path: Path to dataset
        batch_size: Batch size
        split: 'train' or 'val'/'test'
        **kwargs: Additional arguments
    
    Returns:
        DataLoader
    """
    if dataset_name.lower() == 'imagenet':
        loader = ImageNet1KDataLoader(
            data_path=data_path,
            batch_size=batch_size,
            split=split,
            **kwargs
        )
    elif dataset_name.lower() == 'cifar10':
        loader = CifarDataLoader(
            data_path=data_path,
            batch_size=batch_size,
            split=split,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return loader.get_dataloader()