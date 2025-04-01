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
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as label_file:
                for label in label_file:
                    vals = label.strip().split()
                    class_id, x_center, y_center, width, height = map(float, vals)
                    bboxes.append([x_center, y_center, width, height])  
                    labels.append(int(class_id))  
        
        if not bboxes: #Should not be a problem with this dataset
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes.tolist(), labels=labels.tolist())
            image = transformed["image"]
            bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)

        return image, {"boxes": bboxes, "labels": labels}
    
transform = A.Compose(
    [
        A.ChannelDropout(), # Thought is that this will generalize better to shifting wheater conditiotion -> Varying colors
        A.Defocus(p=0.2), # Snow on the lens or poles that are to far/close to be in the focus of the camera
        A.GaussNoise(p=0.2), # Some of the same effects as dropout layers in the network
        A.Illumination(p=0.2), # Shifting light intensity, usual when driving in and out of a forest, snow, etc
        A.RandomBrightnessContrast(p=0.2), # The camera exposure might be of
        A.HorizontalFlip(), # To increase sample size
        A.VerticalFlip(), # To increase sample size
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=["labels"])
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
