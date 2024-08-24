import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pathlib import Path
"""
Contain for setting updata with full func to create dataloader
"""

def create_dataloader(
                    train_transform: transforms.Compose,
                    val_transform: transforms.Compose,
                    test_transform: transforms.Compose,
                    **kwargs,
                ):
    
    train_path = Path(kwargs['dataset']['train'])
    val_path = Path(kwargs['dataset']['val'])
    test_path = Path(kwargs['dataset']['test'])
    
    num_workers = 6
    
    train_data = ImageFolder(
        root= train_path,
        transform= train_transform
    )
    
    val_data = ImageFolder(
        root= val_path,
        transform= val_transform
    )

    test_data = ImageFolder(
        root= test_path,
        transform= test_transform
    )
    
    train_dataloader = DataLoader(
        dataset= train_data,
        batch_size= kwargs['dataset']['batch_size'],
        num_workers=num_workers,
        pin_memory= True,
        persistent_workers= True,
        shuffle= True
    )
    
    val_dataloader = DataLoader(
        dataset= val_data,
        batch_size= kwargs['dataset']['batch_size'],
        num_workers=num_workers,
        pin_memory= True,
        persistent_workers= True,
        shuffle= False
    )

    test_dataloader = DataLoader(
        dataset= test_data,
        batch_size= kwargs['dataset']['batch_size'],
        shuffle= False
    )
    class_names = train_data.classes
    return train_dataloader, val_dataloader, test_dataloader, class_names
