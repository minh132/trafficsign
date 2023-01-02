import numpy as np
import cv2
import torch
import glob as glob
import pandas as pd
import os
import albumentations as A
import time
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
from torch import topk
from model import build_model
device = 'cuda'
# Class names.
sign_names_df = pd.read_csv('../input/signnames.csv')
class_names = sign_names_df.SignName.tolist()
# DataFrame for ground truth.
gt_df = pd.read_csv(
    '../input/GTSRB_Final_Test_GT/GT-final_test.csv', 
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
        '../outputs/model.pth', map_location=device
    )['model_state_dict']
)
def returnCAM(feature_conv, weight_softmax, class_idx):
    '''Tạo class activation maps'''
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
def apply_color_map(CAMs, width, height, orig_image):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.5
        result = cv2.resize(result, (224, 224))
        return result
def visualize_and_save_map(
    result, orig_image, gt_idx=None, class_idx=None, save_name=None
):
    '''Đưa nhãn vào ảnh và lưu lại'''
    if class_idx is not None:
        cv2.putText(
            result, 
            f"Pred: {str(class_names[int(class_idx)])}", (5, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            cv2.LINE_AA
        )
    if gt_idx is not None:
        cv2.putText(
            result, 
            f"GT: {str(class_names[int(gt_idx)])}", (5, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            cv2.LINE_AA
        )
    orig_image = cv2.resize(orig_image, (224, 224))
    img_concat = cv2.hconcat([
        np.array(result, dtype=np.uint8), 
        np.array(orig_image, dtype=np.uint8)
    ])
    if save_name is not None:
        cv2.imwrite(f"../outputs/test_results/CAM_{save_name}.jpg", img_concat)
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
model._modules.get('features').register_forward_hook(hook_feature)
params = list(model.parameters())
weight_softmax = np.squeeze(params[-4].data.cpu().numpy())
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
    ])
counter = 0
all_images = glob.glob('../input/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/*.ppm')
correct_count = 0
frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second. 
# Kiem tra toan bo tap test
for i, image_path in enumerate(all_images):
    image = cv2.imread(image_path)
    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = orig_image.shape
    image_tensor = transform(image=image)['image']
    image_tensor = image_tensor.unsqueeze(0)
    start_time = time.time()
    outputs = model(image_tensor.to(device))
    end_time = time.time()
    probs = F.softmax(outputs).data.squeeze()
    class_idx = topk(probs, 1)[1].int()
    image_name = image_path.split(os.path.sep)[-1]
    gt_idx = gt_df.loc[image_name].ClassId
    #Kiểm tra xem dự đoán có chính xác hay không
    if gt_idx == class_idx:
        correct_count += 1
    CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
    save_name = f"{image_path.split('/')[-1].split('.')[0]}"
    #Lưu lại kết quả
    result = apply_color_map(CAMs, width, height, orig_image)
    visualize_and_save_map(result, orig_image, gt_idx, class_idx, save_name)
print(f"Total number of test images: {len(all_images)}")
print(f"Total correct predictions: {correct_count}")
print(f"Accuracy: {correct_count/len(all_images)*100:.3f}")
