import torch
import torchvision

from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights

def create_model(num_classes=2, size=300, weights_path=None):
    # Load the Torchvision pretrained model.
    model = torchvision.models.detection.ssd300_vgg16(
        # weights=None if weights_path else SSD300_VGG16_Weights.COCO_V1
        weights=None,
        weights_backbone=None
    )
    
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict, strict=False)
    
    # Retrieve the list of input channels. 
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    # List containing number of anchors based on aspect ratios.
    num_anchors = model.anchor_generator.num_anchors_per_location()
    # The classification head.
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    # Image size for transforms.
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model

if __name__ == '__main__':
    weights_path = '/home/jupyter/ee_tree_counting/Models/SSD/weights/ssd300_vgg16_coco-b556d3b4.pth'
    model = create_model(2, 640, weights_path=weights_path)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")