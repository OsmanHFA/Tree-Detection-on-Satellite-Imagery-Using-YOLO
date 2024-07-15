########################################## Old Training Loop ##########################################

# Initialize datasets and data loaders
train_dataset = create_train_dataset(TRAIN_DIR)
valid_dataset = create_valid_dataset(VALID_DIR)
train_loader = create_train_loader(train_dataset, NUM_WORKERS)
valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")

# Initialize the model, optimizer, scaler, and scheduler
model = create_model(num_classes=NUM_CLASSES, weights_path=WEIGHTS_PATH)
model = model.to(DEVICE)
# print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Initialize monitoring metrics
train_loss_list = []
map_50_list = []
map_list = []
metric = MeanAveragePrecision()

# Initialize objects to save the best model and visualize transformed images
save_best_model = SaveBestModel()
if VISUALIZE_TRANSFORMED_IMAGES:
    from custom_utils import show_tranformed_image
    show_tranformed_image(train_loader)

# Training and validation loops
for epoch in range(NUM_EPOCHS):
    train_loss = train(train_loader, model, optimizer, scaler)
    train_loss_list.append(train_loss)
    
    metric_summary = validate(valid_loader, model)
    map_50_list.append(metric_summary['map_50'])
    map_list.append(metric_summary['map'])
    
    print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f} mAP@0.5: {metric_summary['map_50']:.4f} mAP: {metric_summary['map']:.4f}")
    
    # Save the best model and plots
    save_best_model(model, float(metric_summary['map']), epoch, OUT_DIR)
    save_model(epoch, model, optimizer)
    save_loss_plot(OUT_DIR, train_loss_list, x_label='epochs', y_label='train loss', save_name='train_loss')
    save_mAP(OUT_DIR, map_50_list, map_list)

    # scheduler.step()  # Step the learning rate scheduler

print("Training complete.")

def train(train_data_loader, model, optimizer, scaler):
    model.train()
    train_loss_hist = Averager()
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss_hist.send(losses.item())
        prog_bar.set_description(desc=f"Epoch {epoch+1} Loss: {train_loss_hist.value:.4f}")
        
        # Clear memory
        del images, targets, loss_dict, losses
        torch.cuda.empty_cache()
    
    return train_loss_hist.value

def validate(valid_data_loader, model):
    print('Validating')
    model.eval()
    val_loss_hist = Averager()
    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            # Get both losses and outputs
            loss_dict = model(images, targets)
            outputs = model(images)
        print(f"loss_dict structure: {loss_dict}")
        # Calculate total loss
        # Assuming loss_dict is a list of dictionaries, each containing a 'loss' key
        losses = sum(item['loss'] for item in loss_dict if 'loss' in item)
        val_loss_hist.send(losses.item())
        
        # For mAP calculation using Torchmetrics.
        for j in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[j]['boxes'].detach().cpu()
            true_dict['labels'] = targets[j]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[j]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[j]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[j]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
    
    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    
    return val_loss_hist.value, metric_summary

########################################## Old Dataset.py ##########################################
import torch
import cv2
import numpy as np
import os
import glob as glob
from xml.etree import ElementTree as et
from config import CLASSES, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform
import matplotlib.pyplot as plt

# Utility function to display an image using matplotlib
def show_image(image, title="Image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

class CustomDataset(Dataset):
    def __init__(self, dir_path, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []
        
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.dir_path, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        while True:
            try:
                image_name = self.all_images[idx]
                image_path = os.path.join(self.dir_path, image_name)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Failed to load image at {image_path}")
                    idx = (idx + 1) % len(self.all_images)
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
                image /= 255.0

                annot_filename = os.path.splitext(image_name)[0] + '.xml'
                annot_file_path = os.path.join(self.dir_path, annot_filename)
                
                boxes = []
                labels = []
                tree = et.parse(annot_file_path)
                root = tree.getroot()
                
                image_width = image.shape[1]
                image_height = image.shape[0]
                
                for member in root.findall('object'):
                    labels.append(self.classes.index(member.find('name').text))
                    
                    xmin = int(member.find('bndbox').find('xmin').text) / image_width
                    xmax = int(member.find('bndbox').find('xmax').text) / image_width
                    ymin = int(member.find('bndbox').find('ymin').text) / image_height
                    ymax = int(member.find('bndbox').find('ymax').text) / image_height
                    
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                
                if len(boxes) != len(labels):
                    print(f"Mismatch between boxes and labels at {annot_file_path}")
                    idx = (idx + 1) % len(self.all_images)
                    continue
                
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.as_tensor(boxes, dtype=torch.float32)
                iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
                labels = torch.as_tensor(labels, dtype=torch.int64)

                target = {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd, "image_id": torch.tensor([idx])}

                if self.transforms:
                    sample = self.transforms(image=image, bboxes=target['boxes'], labels=labels)
                    image = sample['image']
                    target['boxes'] = torch.Tensor(sample['bboxes'])

                return image, target
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                idx = (idx + 1) % len(self.all_images)
                continue

    def __len__(self):
        return len(self.all_images)

def create_train_dataset(DIR):
    train_dataset = CustomDataset(
        DIR, CLASSES, get_train_transform()
    )
    return train_dataset

def create_valid_dataset(DIR):
    valid_dataset = CustomDataset(
        DIR, CLASSES, get_valid_transform()
    )
    return valid_dataset

def create_test_dataset(DIR):
    test_dataset = CustomDataset(
        DIR, CLASSES, get_valid_transform()
    )
    return test_dataset

def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return train_loader

def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return valid_loader

def create_test_loader(test_dataset, num_workers=0):
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return test_loader

if __name__ == '__main__':
    dataset = CustomDataset(
        '/home/jupyter/ee_tree_counting/Data/Combined Dataset XML/train', CLASSES 
    )
    print(f"Number of training images: {len(dataset)}")
    
    def visualize_sample(image, target, output_dir='visualisations/Dataset'):
        os.makedirs(output_dir, exist_ok=True)
        image_copy = (image.copy() * 255).astype(np.uint8)
        
        # Display the original image
        show_image(image_copy, title="Original Image")

        # Draw bounding boxes
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            x1, y1, x2, y2 = box
            x1 = int(x1 * image_copy.shape[1])
            y1 = int(y1 * image_copy.shape[0])
            x2 = int(x2 * image_copy.shape[1])
            y2 = int(y2 * image_copy.shape[0])
            
            # Debug: Print box coordinates
            print(f"Box {box_num}: ({x1}, {y1}), ({x2}, {y2})")
            
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save and display the annotated image
        image_name = f"{target['image_id'].item()}.jpg"
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
        show_image(image_copy, title="Annotated Image")
        
    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)

        

########################################## Debugging Code ##########################################

dataset = CustomDataset(
    '/home/jupyter/ee_tree_counting/Data/Combined Dataset XML No Aug/train', CLASSES 
)
print(f"Number of training images: {len(dataset)}")

def show_image_with_annotations(image, boxes, labels):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        x1, y1, x2, y2 = box
        x1 = int(x1 * image.shape[1])
        y1 = int(y1 * image.shape[0])
        x2 = int(x2 * image.shape[1])
        y2 = int(y2 * image.shape[0])
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', lw=2))
        plt.text(x1, y1 - 5, CLASSES[label], color='red', fontsize=12, backgroundcolor='white')
    plt.axis('off')
    plt.show()

def inspect_dataset_samples(dataset, num_samples=1):
    for i in range(num_samples):
        image, target = dataset[i]
        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()
        show_image_with_annotations(image, boxes, labels)

# Assuming dataset is already defined
inspect_dataset_samples(dataset)

import matplotlib.pyplot as plt

def show_image_with_annotations(image, boxes, labels):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    height, width, _ = image.shape
    
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        
        # Ensure the box coordinates are in the correct range
        if box.max() > 1:  # Assuming box values are not normalized
            x1, y1, x2, y2 = box
        else:  # If normalized, convert to absolute pixel values
            x1, y1, x2, y2 = box * [width, height, width, height]
        
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', lw=2))
        plt.text(x1, y1 - 5, CLASSES[label], color='red', fontsize=12, backgroundcolor='white')
    
    plt.axis('off')
    plt.show()

def inspect_model_predictions(model, dataset, num_samples=1):
    model.eval()
    for i in range(num_samples):
        image, target = dataset[i]
        
        print(f"Image {i}: shape {image.shape}, target {target}")
        
        # Ensure image tensor is correctly shaped and normalized
        if image.max() > 1:
            image = image / 255.0
        
        image_tensor = image.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device
        with torch.no_grad():
            output = model(image_tensor)[0]  # Get the first (and only) prediction

        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        print(f"Predicted boxes: {boxes}")
        print(f"Predicted labels: {labels}")
        
        show_image_with_annotations(image.permute(1, 2, 0).numpy(), boxes, labels)

# Assuming dataset is already defined and model is loaded
inspect_model_predictions(model, valid_dataset)

########################################## Old Eval.py ##########################################


import torch

from tqdm import tqdm
from config import DEVICE, NUM_CLASSES, NUM_WORKERS, WEIGHTS_PATH
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from RetinaNet import create_model
from Datasets import create_valid_dataset, create_valid_loader  # Assuming these functions exist
from Datasets import create_test_dataset, create_test_loader  # Updated import


# Evaluation function
def validate(valid_data_loader, model):
    model.eval()

    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary

if __name__ == '__main__':
    # Load the best model and trained weights.
    model = create_model(num_classes=NUM_CLASSES, weights_path=WEIGHTS_PATH)
    # model = model.to(DEVICE)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    test_dataset = create_valid_dataset('/home/jupyter/ee_tree_counting/Data/Combined Dataset XML/test')  # Change to your test directory path
    test_loader = create_valid_loader(test_dataset, num_workers=NUM_WORKERS)

    metric_summary = validate(test_loader, model)
    print(f"mAP_50: {metric_summary['map_50']*100:.3f}")
    print(f"mAP_50_95: {metric_summary['map']*100:.3f}")
