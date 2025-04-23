# rgb_dataloader.py (or wherever your dataloader code is)

import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RGBDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_width = 1920, img_height = 1208, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_width = img_width  
        self.img_height = img_height
        self.transform = transform
        
        self.image_filenames = sorted([f for f in os.listdir(img_dir) if f.endswith(".PNG") or f.endswith(".png")])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # Load Image
        img_name = self.image_filenames[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
             raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        original_h, original_w, _ = image.shape

        label_path = os.path.join(self.label_dir, img_name.replace(".PNG", ".txt").replace(".png", ".txt"))
        bboxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as label_file:
                for label in label_file:
                    vals = label.strip().split()
                   
                    if len(vals) == 5:
                        class_id, x_center, y_center, width, height = map(float, vals)

                        abs_x_center = x_center * original_w
                        abs_y_center = y_center * original_h
                        abs_width = width * original_w
                        abs_height = height * original_h

                        x_min = int(abs_x_center - abs_width / 2)
                        y_min = int(abs_y_center - abs_height / 2)
                        x_max = int(abs_x_center + abs_width / 2)
                        y_max = int(abs_y_center + abs_height / 2)

                        bboxes.append([x_min, y_min, x_max, y_max])
                        
                        labels.append(int(1))
                
                    else:
                        print(f"Skipping malformed line in {label_path}: {label.strip()}")

        if not bboxes:
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": bboxes_tensor, "labels": labels_tensor}

        if self.transform:
            transformed = self.transform(image=image, bboxes=target["boxes"].tolist(), labels=target["labels"].tolist())
            image = transformed["image"]
           
            target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32) if transformed["bboxes"] else torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64) if transformed["labels"] else torch.zeros((0,), dtype=torch.int64)

        return image, target, img_name


transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ToFloat(max_value=255.0),
        A.Normalize(normalization="min_max", max_pixel_value=1.0, p=1.0), 
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=["labels"])
)

valid_transform = A.Compose(
    [
        A.ToFloat(max_value=255.0),
        A.Normalize(normalization="min_max", max_pixel_value=1.0, p=1.0),
        ToTensorV2(),
    ],
     bbox_params=A.BboxParams(format='pascal_voc', label_fields=["labels"]) 
)

def collate_fn_save(batch):
    """
    Collate function to handle images, targets, and filenames.
    Assumes batch_size=1 for simplicity in saving individual files.
    """
    if not batch:
        return None, None, None
    images, targets, filenames = zip(*batch)
    images = torch.stack(images, dim=0) if len(images) > 1 else list(images)
    return images, list(targets), list(filenames)


rgb_testset = RGBDataset(
    img_dir="/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Poles/rgb/images/test", 
    label_dir="/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Poles/rgb/labels/valid",
    transform=valid_transform) 

rgb_testloader = DataLoader(
    rgb_testset,
    batch_size=1, # IMPORTANT: Use batch_size=1 for easy file saving per image
    shuffle=False, 
    collate_fn=collate_fn_save, 
    num_workers=1)

lidar_testset = RGBDataset(
    img_dir="/home/mariumre/Documents/SnowPoleDetection/Poles/lidar/combined_color/test", 
    label_dir="/home/mariumre/Documents/SnowPoleDetection/Poles/rgb/labels/valid",
    transform=valid_transform) 

lidar_testloader = DataLoader(
    lidar_testset,
    batch_size=1, # IMPORTANT: Use batch_size=1 for easy file saving per image
    shuffle=False,
    collate_fn=collate_fn_save, 
    num_workers=4)

