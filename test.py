import numpy as np
import cv2
import torch
import glob as glob
import pandas as pd
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
from torch import topk
from model import build_model
device = 'cuda'
# Class names.
sign_names_df = pd.read_csv('..\\data\\signnames.csv')
class_names = sign_names_df.SignName.tolist()
# DataFrame for ground truth.
gt_df = pd.read_csv(
    '..\\data\\GT-final_test.csv', 
    delimiter=';'
)
gt_df = gt_df.set_index('Filename', drop=True)
# Initialize model, switch to eval model, load trained weights.
model = build_model(
    pretrained=False,
    fine_tune=False, 
    num_classes=43
).to(device)
model = model.eval()
model.load_state_dict(
    torch.load(
        '..\\output\\model.pth', map_location=device
    )['model_state_dict']
)
