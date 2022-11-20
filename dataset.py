import torch
import numpy as np
import albumentations as A
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from albumentations.pytorch import ToTensorV2
root='trafficsign\data\Final_Training\Images'
valid_split=0.1
resize=224
batch_size=64
# Training transforms.
class TrainTransforms:
    def __init__(self, resize):
        self.transforms = A.Compose([
            A.Resize(resize, resize),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                ),
            ToTensorV2()
        ])
    
    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']
# Validation transforms.
class ValidTransforms:
    def __init__(self, resize):
        self.transforms = A.Compose([
            A.Resize(resize, resize),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                ),
            ToTensorV2()
        ])
    
    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']
def get_datasets():
    
    dataset = datasets.ImageFolder(
        root, 
        transform=(TrainTransforms(resize))
    )
    dataset_test = datasets.ImageFolder(
        root, 
        transform=(ValidTransforms(resize))
    )
    dataset_size = len(dataset)
    
    valid_size = int(valid_split*dataset_size)

    indices = torch.randperm(len(dataset)).tolist()

    dataset_train = Subset(dataset, indices[:-valid_size])
    dataset_valid = Subset(dataset_test, indices[-valid_size:])
    return dataset_train, dataset_valid, dataset.classes
def get_data_loaders(dataset_train, dataset_valid):
    
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, 
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, 
    )
    return train_loader, valid_loader