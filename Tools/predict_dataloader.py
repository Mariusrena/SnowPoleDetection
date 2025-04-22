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
        self.img_width = img_width  # Note: These seem unused if transform resizes
        self.img_height = img_height # Note: These seem unused if transform resizes
        self.transform = transform
        # Ensure consistent sorting for reproducibility if needed, otherwise listdir order can vary
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

        # Load Label (for training/validation metrics if needed, not strictly required for prediction saving)
        label_path = os.path.join(self.label_dir, img_name.replace(".PNG", ".txt").replace(".png", ".txt"))
        bboxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as label_file:
                for label in label_file:
                    vals = label.strip().split()
                    # Ensure file is not empty or malformed
                    if len(vals) == 5:
                        class_id, x_center, y_center, width, height = map(float, vals)

                        # Convert to absolute pixel values
                        abs_x_center = x_center * original_w
                        abs_y_center = y_center * original_h
                        abs_width = width * original_w
                        abs_height = height * original_h

                        # Convert from (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
                        x_min = int(abs_x_center - abs_width / 2)
                        y_min = int(abs_y_center - abs_height / 2)
                        x_max = int(abs_x_center + abs_width / 2)
                        y_max = int(abs_y_center + abs_height / 2)

                        bboxes.append([x_min, y_min, x_max, y_max])
                        # *** IMPORTANT ***
                        # Your requested output format starts with class '0'.
                        # Your original code forces labels to '1'.
                        # Decide which class ID you actually want. Sticking to '1' based on your code:
                        labels.append(int(1))
                        # If you need class 0 in the output file, change the above line to:
                        # labels.append(int(class_id)) # Or simply labels.append(0) if it's always 0
                    else:
                        print(f"Skipping malformed line in {label_path}: {label.strip()}")


        # Ensure tensors are created even if no labels exist
        if not bboxes:
            # Use float32 for bounding boxes as transforms might output float
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            # Use float32 for bounding boxes
            bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": bboxes_tensor, "labels": labels_tensor} # Keep target separate

        if self.transform:
            # Pass bboxes as list of lists, labels as list for albumentations
            transformed = self.transform(image=image, bboxes=target["boxes"].tolist(), labels=target["labels"].tolist())
            image = transformed["image"]
            # Ensure bboxes and labels are tensors after transform
            target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32) if transformed["bboxes"] else torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64) if transformed["labels"] else torch.zeros((0,), dtype=torch.int64)

        # Return image, target dictionary, and the original filename
        return image, target, img_name

# Define transforms (keep your existing ones)
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ToFloat(max_value=255.0), # Use float value here
        #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Example using ImageNet stats
        # Or keep your min_max normalization if preferred:
        A.Normalize(normalization="min_max", max_pixel_value=1.0, p=1.0), # Apply after ToFloat
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=["labels"])
)

valid_transform = A.Compose(
    [
        A.ToFloat(max_value=255.0),
        #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.Normalize(normalization="min_max", max_pixel_value=1.0, p=1.0),
        ToTensorV2(),
    ],
     bbox_params=A.BboxParams(format='pascal_voc', label_fields=["labels"]) # Add bbox_params here too
)

def collate_fn_save(batch):
    """
    Collate function to handle images, targets, and filenames.
    Assumes batch_size=1 for simplicity in saving individual files.
    """
    if not batch:
        return None, None, None
    # Batch is a list of tuples: [(img1, target1, fname1), (img2, target2, fname2), ...]
    images, targets, filenames = zip(*batch)
    # If batch_size=1, images will be a tuple with one tensor, targets a tuple with one dict, filenames a tuple with one string.
    # Stack images if batch_size > 1, otherwise just use the single image tensor (but keep in a list for consistency)
    images = torch.stack(images, dim=0) if len(images) > 1 else list(images) # Return list for consistency with model input expectation
    return images, list(targets), list(filenames) # Keep targets and filenames as lists


# --- Dataloader Instantiation ---
# Use the validation set for generating predictions
# Use batch_size=1 to process and save one image at a time easily
# Use the modified collate function

# Create the validation dataset *without* augmentations that change geometry drastically for prediction
# (like flips, unless you want predictions on flipped images too)
# Usually, only normalization and ToTensor are used for validation/prediction.
rgb_testset = RGBDataset(
    img_dir="/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Poles/rgb/images/test", # Or use train/test split as needed
    label_dir="/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Poles/rgb/labels/valid",
    transform=valid_transform) # Use the simple validation transform

rgb_testloader = DataLoader(
    rgb_testset,
    batch_size=1, # IMPORTANT: Use batch_size=1 for easy file saving per image
    shuffle=False, # No need to shuffle for prediction saving
    collate_fn=collate_fn_save, # Use the modified collate function
    num_workers=1 # Adjust based on your system
    )

# Keep train loader as is if needed for training
rgb_trainset = RGBDataset(
    img_dir="/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Poles/rgb/images/train",
    label_dir="/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Poles/rgb/labels/train",
    transform=transform)
rgb_trainloader = DataLoader(rgb_trainset, batch_size=1, shuffle=True, collate_fn=collate_fn_save) # Can use same collate fn