import cv2
from pathlib import Path

# Draw bounding boxes from YOLO format
def draw_bounding_box(image_path, label_path):
    
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Read labels file for specified image
    with open(label_path, "r") as label_file:
        labels = label_file.readlines()

    for label in labels:
        values = label.strip().split()
        class_id = int(values[0])
        x_center, y_center, bb_width, bb_height = map(float, values[1:])

        # Convert YOLO format to pixel values
        x1 = int((x_center - bb_width / 2) * image_width)
        y1 = int((y_center - bb_height / 2) * image_height)
        x2 = int((x_center + bb_width / 2) * image_width)
        y2 = int((y_center + bb_height / 2) * image_height)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Snow Pole BBox", image)

    key = cv2.waitKey(0)
    return key
    


if __name__ == "__main__":

    # Iterates through a folder, using the filename to find linked label in another folder
    # Use ENTER to see next image and ESC to quit. 

    image_paths = Path("/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Poles/lidar/combined_color/train")
    
    label_paths = "/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Poles/lidar/labels/train"

    for image_path in image_paths.iterdir():
        try:
            label_path = "".join([label_paths, "/", image_path.stem, ".txt"])
            key = draw_bounding_box(image_path, label_path) 
            
            if key == 13:    
                cv2.destroyAllWindows()
            elif key == 27:
                break
        except:
            print(f"No label found for {image_path.stem}")

    