import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as transforms
import albumentations as A

class RGBDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_width = 1920, img_height = 1208, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_width = img_width
        self.img_height = img_height
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(img_dir) if f.endswith(".PNG") or f.endswith(".png")]
    
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # Load Image
        img_name = self.image_filenames[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        original_h, original_w, _ = image.shape

        # Load Label
        label_path = os.path.join(self.label_dir, img_name.replace(".PNG", ".txt").replace(".png", ".txt"))
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as label_file:
                for label in label_file:
                    vals = label.strip().split()
                    class_id, x_center, y_center, width, height = map(float, vals[::])
                    
                    bboxes.append([class_id, x_center, y_center, width, height])
        
        # Checks if there is a bbox in the image, should not be an issue with this dataset
        bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 5))

        # Resize image and adjust bounding boxes
        if self.transform:
            image = self.transform(image)
        
        return image, bboxes


transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.ToTensor(),
    ],
    bbox_params=A.BboxParams(format='coco')
)   


rgb_dataset = RGBDataset(
    img_dir="/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Poles/rgb/images/train",
    label_dir="/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Poles/rgb/labels/train",
    transform=transform)


def collate_fn(batch):
    images, labels = zip(*batch)  # Unzip batch
    images = torch.stack(images, dim=0)  # Stack images normally
    return images, list(labels)  # Keep labels as a list (variable size)

rgb_dataloader = DataLoader(rgb_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

if __name__ == "__main__":
    images, labels = next(iter(rgb_dataloader))
    print(images.shape)  # (batch_size, channels (3), width, height)
    print(labels)        # Bounding boxes
