import torch

BATCH_SIZE = 4 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 100 # Number of epochs to train for.
NUM_WORKERS = 0 # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

WEIGHTS_PATH = '/home/jupyter/ee_tree_counting/Models/SSD/weights/ssd300_vgg16_coco-b556d3b4.pth'


# Training images and XML files directory.
TRAIN_DIR = '/home/jupyter/ee_tree_counting/Data/Final Tree Dataset XML/train'
# Validation images and XML files directory.
VALID_DIR = '/home/jupyter/ee_tree_counting/Data/Final Tree Dataset XML/valid'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'tree'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after creating the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = True

# Location to save model and plots.
OUT_DIR = 'outputs'