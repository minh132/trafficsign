import torch
import numpy as np
import albumentations as A
import pydoc
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from albumentations.pytorch import ToTensorV2
root='..\\data\\Final_Training\\Images'
valid_split=0.1
resize=224
batch_size=64
MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]
class Transforms:
    """_summary
    Chuẩn hóa dữ liệu 
    """
    def __init__(self, resize):
        """Chuẩn hóa các giá trị bằng cách sử dụng giá trị trung bình và độ lệch chuẩn của ImageNet và chuyển đổi hình ảnh thành tensor"""
        self.transforms = A.Compose([
            A.Resize(resize, resize),
            A.Normalize(
                mean=MEAN,
                std=STD,
                ),
            ToTensorV2()
        ])
    
    def __call__(self, img):
        """Truyền hình ảnh qua self.transforms"""
        return self.transforms(image=np.array(img))['image']

def get_datasets():
    """
    Chuẩn bị cho tập datasets
    Trả về dữ liệu tập train và tập test cùng với tên class
    """
    dataset = datasets.ImageFolder(
        root, 
        transform=(Transforms(resize))
    )
    dataset_test = datasets.ImageFolder(
        root, 
        transform=(Transforms(resize))
    )
    dataset_size = len(dataset)
    valid_size = int(valid_split*dataset_size)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = Subset(dataset, indices[:-valid_size])
    dataset_valid = Subset(dataset_test, indices[-valid_size:])
    return dataset_train, dataset_valid, dataset.classes
def get_data_loaders(dataset_train, dataset_valid):
    """
    Chuẩn bị cho dataloaders
    
    Trả về tập dữ liệu train_loader và valid_loader
    """
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, 
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, 
    )
    return train_loader, valid_loader