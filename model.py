import torchvision.models as models
import torch.nn as nn
def build_model(pretrained=True, fine_tune=False, num_classes=10):
    """
    Sử dụng mô hình pretrained  MobileNetV3 Large
    
    Hàm trên sẽ trả về  mô hình dựa trên việc ta có muốn tải các trọng số đã được đào tạo trước hay không và cũng có muốn tinh chỉnh tất cả các lớp hay không.
    """
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.mobilenet_v3_large(pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    model.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes)
    return model