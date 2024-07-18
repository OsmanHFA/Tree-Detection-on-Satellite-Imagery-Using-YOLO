import torchvision
import torch
from functools import partial
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def create_model(num_classes=2, weights_path=None):
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=None if weights_path else RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    )
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    return model

if __name__ == '__main__':
    weights_path = '/home/jupyter/ee_tree_counting/Models/RetinaNet/weights/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth'
    model = create_model(4, weights_path=weights_path)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")