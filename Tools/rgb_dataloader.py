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
                    class_id = int(vals[0])
                    x_center, y_center, width, height = map(float, vals[1:])
                    
                    # Convert to absolute pixel values, easier to work with transformations etc
                    x_center *= original_w
                    y_center *= original_h
                    width *= original_w
                    height *= original_h
                    
                    # Convert from (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)
                    
                    bboxes.append([x_min, y_min, x_max, y_max, class_id])
        
        # Checks if there is a bbox in the image, should not be an issue with this dataset
        bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 5))

        # Resize image and adjust bounding boxes
        if self.transform:
            image = self.transform(image)
        
        return image, bboxes


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((1920, 1208)),
    transforms.ToTensor()
])



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
