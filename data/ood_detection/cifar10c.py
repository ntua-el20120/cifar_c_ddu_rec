import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CIFAR10C(Dataset):
    """
    A PyTorch Dataset for CIFAR10-C.
    """

    def __init__(self, 
                 root, 
                 corruption='gaussian_noise', 
                 severity=1, 
                 transform=None):
        """
        Args:
            root (str): Path to the CIFAR-10-C folder containing .npy files
            corruption (str): Which corruption type to load (e.g. 'gaussian_noise').
            severity (int): 1 to 5, which severity level of corruption.
            transform (callable, optional): Optional transform to be applied
                on a sample (image).
        """
        super().__init__()
        self.root = root
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        
        # Load .npy for the specified corruption
        data_path = os.path.join(root, f"{corruption}.npy")
        label_path = os.path.join(root, "labels.npy")
        
        self.data = np.load(data_path)  # shape (50000, 32, 32, 3)
        self.targets = np.load(label_path)  # shape (50000,)
        
        # Each severity is a block of 10,000 images
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        
        # Slice just the portion corresponding to that severity
        self.data = self.data[start_idx:end_idx]
        self.targets = self.targets[start_idx:end_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        # Convert to PIL or tensor, etc. so that transforms can be applied
        img = img.astype(np.uint8)  # ensure it's in uint8
        img = transforms.functional.to_pil_image(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


def get_cifar10_c_loader(corruption,
                         severity=1,
                         batch_size=128,
                         num_workers=4,
                         pin_memory=False):
    """
    Returns a DataLoader for the specified corruption and severity in CIFAR10-C.
    
    Args:
        root (str): Path to the folder containing CIFAR-10-C files.
        corruption (str): e.g. 'gaussian_noise', 'motion_blur', etc.
        severity (int): 1 to 5
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): Whether to use pinned memory in DataLoader.
    
    Returns:
        DataLoader: An iterator over the specified CIFAR10-C corruption/severity.
    """
    # Define the same normalization as for standard CIFAR-10
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    path = "/home/ubuntu/workspace/recreation/DDU/data/CIFAR-10-C"

    dataset = CIFAR10C(root=path,
                       corruption=corruption,
                       severity=severity,
                       transform=transform)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,  # Typically no need to shuffle a test set
                        num_workers=num_workers,
                        pin_memory=pin_memory)
    return loader
