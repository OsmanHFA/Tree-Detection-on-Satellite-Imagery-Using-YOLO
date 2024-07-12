import torch
BATCH_SIZE = 16
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 100
NUM_WORKERS = 4 # Number of parallel workers for data loading.
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Training images and XML files directory.
TRAIN_DIR = '/home/jupyter/ee_tree_counting/Data/Combined Dataset XML/train'
# Validation images and XML files directory.
VALID_DIR = '/home/jupyter/ee_tree_counting/Data/Combined Dataset XML/valid'
# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'tree'
]
NUM_CLASSES = len(CLASSES)
# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False
# Location to save model and plots.
OUT_DIR = 'outputs'