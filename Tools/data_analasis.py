import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
import warnings

""" 
    Disclaimer: Solution provided purely by Gemini 2.5 Pro!
    Code was used to analyze aspect ratios and anchor sizes when model wouldnt learn.
"""

# Suppress potential divide-by-zero warnings for aspect ratio if height is 0
warnings.filterwarnings("ignore", category=RuntimeWarning)

class RGBDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_width = 1920, img_height = 1208):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_width = img_width 
        self.img_height = img_height
        self.image_filenames = [f for f in os.listdir(img_dir) if f.endswith(".PNG") or f.endswith(".png")]
        print(f"Found {len(self.image_filenames)} images in {img_dir}")
        if not self.image_filenames:
             raise FileNotFoundError(f"No images found in directory: {img_dir}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # Load Image
        img_name = self.image_filenames[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}. Skipping.")
            # Return dummy data 
            return torch.zeros((3, 100, 100)), {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)}

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        original_h, original_w, _ = image.shape

        # Load Label
        label_path = os.path.join(self.label_dir, img_name.replace(".PNG", ".txt").replace(".png", ".txt"))
        bboxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as label_file:
                for label_line in label_file: 
                    vals = label_line.strip().split()
                    if len(vals) != 5:
                         print(f"Warning: Skipping invalid line in {label_path}: {label_line.strip()}")
                         continue
                    try:
                        class_id, x_center, y_center, width, height = map(float, vals)
                    except ValueError:
                        print(f"Warning: Skipping line with non-float values in {label_path}: {label_line.strip()}")
                        continue

                    # Convert relative YOLO format to absolute pixel values
                    abs_x_center = x_center * original_w
                    abs_y_center = y_center * original_h
                    abs_width = width * original_w
                    abs_height = height * original_h

                    # Convert from (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
                    x_min = abs_x_center - abs_width / 2
                    y_min = abs_y_center - abs_height / 2
                    x_max = abs_x_center + abs_width / 2
                    y_max = abs_y_center + abs_height / 2

                    # Clip coordinates to image bounds to prevent issues with slight float inaccuracies
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(original_w, x_max)
                    y_max = min(original_h, y_max)

                    # Ensure box has positive width and height after conversion/clipping
                    if x_max > x_min and y_max > y_min:
                        bboxes.append([x_min, y_min, x_max, y_max])
                        labels.append(int(class_id))
                    else:
                         print(f"Warning: Degenerate box after conversion in {label_path} from line '{label_line.strip()}'. Original dims: w={abs_width}, h={abs_height}. Coords: [{x_min}, {y_min}, {x_max}, {y_max}]")

        target = {} 
        if not bboxes:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        else:
            target["boxes"] = torch.tensor(bboxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)

        # Ensure image is a tensor even if transform is None
        if not isinstance(image, torch.Tensor):
             # Basic conversion: HWC -> CHW, Normalize to [0, 1]
             image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, target 




TRAIN_IMG_DIR = "/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Poles/rgb/images/train"
TRAIN_LABEL_DIR = "/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Poles/rgb/labels/train"
NUM_CLUSTERS_RATIOS = 6  # How many aspect ratio clusters to find (suggests anchor ratios)
NUM_CLUSTERS_SIZES = 9   # How many width/height clusters to find (suggests anchor shapes/sizes)


print("Initializing Dataset for Analysis (transform=None)...")

analysis_dataset = RGBDataset(
    img_dir=TRAIN_IMG_DIR,
    label_dir=TRAIN_LABEL_DIR,
    transform=None  
)

if len(analysis_dataset) == 0:
    print("Dataset is empty. Exiting analysis.")
    exit()

print(f"Starting analysis on {len(analysis_dataset)} training images...")

widths = []
heights = []
areas = []
aspect_ratios = []

# Iterate through the dataset
for i in tqdm(range(len(analysis_dataset)), desc="Analyzing dataset"):
    try:
        image, target = analysis_dataset[i]
        boxes = target['boxes'] # Boxes are [x_min, y_min, x_max, y_max] format

        if boxes.shape[0] > 0: # If there are bounding boxes for this image
             boxes = boxes.float()
             w = boxes[:, 2] - boxes[:, 0]
             h = boxes[:, 3] - boxes[:, 1]

             # Filter out potential zero width/height boxes before calculation
             valid_mask = (w > 0) & (h > 0)
             w = w[valid_mask]
             h = h[valid_mask]

             if len(w) > 0: # If valid boxes remain
                  widths.extend(w.tolist())
                  heights.extend(h.tolist())
                  areas.extend((w * h).tolist())
                  # Calculate aspect ratio (w/h), handle h=0 (already filtered by valid_mask)
                  ratios = w / h
                  aspect_ratios.extend(ratios.tolist())

    except Exception as e:
        print(f"\nError processing item index {i}: {e}")
        continue


if not widths:
    print("No valid bounding boxes found in the dataset labels. Cannot perform analysis.")
    exit()

print(f"\nAnalysis Complete. Found {len(widths)} valid bounding boxes.")

widths = np.array(widths)
heights = np.array(heights)
areas = np.array(areas)
aspect_ratios = np.array(aspect_ratios)


print("\n--- Summary Statistics ---")
print(f"Widths:  Min={np.min(widths):.2f}, Max={np.max(widths):.2f}, Mean={np.mean(widths):.2f}, Median={np.median(widths):.2f}")
print(f"Heights: Min={np.min(heights):.2f}, Max={np.max(heights):.2f}, Mean={np.mean(heights):.2f}, Median={np.median(heights):.2f}")
print(f"Areas:   Min={np.min(areas):.2f}, Max={np.max(areas):.2f}, Mean={np.mean(areas):.2f}, Median={np.median(areas):.2f}")
print(f"Ratios:  Min={np.min(aspect_ratios):.2f}, Max={np.max(aspect_ratios):.2f}, Mean={np.mean(aspect_ratios):.2f}, Median={np.median(aspect_ratios):.2f}")

# --- Plotting ---
plt.figure(figsize=(20, 10))

# Width Histogram
plt.subplot(2, 3, 1)
plt.hist(widths, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Bounding Box Widths')
plt.xlabel('Width (pixels)')
plt.ylabel('Frequency')

# Height Histogram
plt.subplot(2, 3, 2)
plt.hist(heights, bins=50, color='lightcoral', edgecolor='black')
plt.title('Distribution of Bounding Box Heights')
plt.xlabel('Height (pixels)')
plt.ylabel('Frequency')

# Area Histogram
plt.subplot(2, 3, 3)
plt.hist(areas, bins=50, color='lightgreen', edgecolor='black')
plt.title('Distribution of Bounding Box Areas')
plt.xlabel('Area (pixels^2)')
plt.ylabel('Frequency')
plt.yscale('log') # Use log scale if areas vary wildly

# Aspect Ratio Histogram
plt.subplot(2, 3, 4)
plt.hist(aspect_ratios, bins=50, color='gold', edgecolor='black')
plt.title('Distribution of Bounding Box Aspect Ratios (W/H)')
plt.xlabel('Aspect Ratio')
plt.ylabel('Frequency')

# Width vs Height Scatter Plot
plt.subplot(2, 3, 5)
plt.scatter(widths, heights, alpha=0.1, s=10) # Use low alpha/size for many points
plt.title('Bounding Box Width vs. Height')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.grid(True)



# Cluster Aspect Ratios
print(f"\n--- K-Means Clustering for Aspect Ratios (k={NUM_CLUSTERS_RATIOS}) ---")
kmeans_ratios = KMeans(n_clusters=NUM_CLUSTERS_RATIOS, random_state=42, n_init=10).fit(aspect_ratios.reshape(-1, 1))
suggested_ratios = sorted(kmeans_ratios.cluster_centers_.flatten())
print(f"Suggested Aspect Ratios (centroids): {[f'{r:.2f}' for r in suggested_ratios]}")
print("-> Use these values to guide your `aspect_ratios` parameter.")
# Add K-Means results to Aspect Ratio plot
plt.subplot(2, 3, 4) # Re-select the aspect ratio plot
for center in kmeans_ratios.cluster_centers_:
    plt.axvline(center[0], color='red', linestyle='--', linewidth=1, label=f'{center[0]:.2f}' if center==kmeans_ratios.cluster_centers_[0] else "")
plt.legend(['K-Means Centroids'])


# Cluster Width/Height pairs 
print(f"\n--- K-Means Clustering for Width/Height Pairs (k={NUM_CLUSTERS_SIZES}) ---")
wh_pairs = np.vstack((widths, heights)).T
kmeans_wh = KMeans(n_clusters=NUM_CLUSTERS_SIZES, random_state=42, n_init=10).fit(wh_pairs)
suggested_wh = kmeans_wh.cluster_centers_

suggested_wh = sorted(suggested_wh, key=lambda x: x[0] * x[1])
print("Suggested Width/Height Pairs (centroids):")
for w_cent, h_cent in suggested_wh:
    print(f"  W: {w_cent:.1f}, H: {h_cent:.1f} (Area: {w_cent*h_cent:.1f}, Ratio: {w_cent/h_cent:.2f})")
print("-> These pairs give an idea of common absolute sizes/shapes.")
# Add K-Means centers to Width/Height scatter plot
plt.subplot(2, 3, 5) # Re-select the scatter plot
plt.scatter(kmeans_wh.cluster_centers_[:, 0], kmeans_wh.cluster_centers_[:, 1],
            marker='X', s=100, c='red', edgecolors='black', label='K-Means Centroids')
plt.legend(['Data points', 'K-Means Centroids'])


# Display plots
plt.tight_layout()
plt.show()


print("\n--- How to Use This Analysis ---")
print("1. Sizes: Look at the Width, Height, and Area histograms.")
print("   - Determine the range of object sizes present.")
print("   - Choose `anchor_sizes` that cover this range across your FPN levels.")
print("   - Example: If median area is X, ensure some anchors have sqrt(X) as a base size.")
print("2. Ratios: Look at the Aspect Ratio histogram and K-Means centroids.")
print("   - Identify the most common aspect ratios (peaks in histogram, K-Means centers).")
print("   - Set your `aspect_ratios` parameter to include these common ratios.")
print("   - Example: If K-Means suggests [0.50, 1.00, 1.80], use `aspect_ratios=((0.5, 1.0, 1.8),)` (adjust per level if needed).")
print("3. Width vs Height Plot: Shows the variety of shapes. Helps confirm if ratios seem reasonable.")