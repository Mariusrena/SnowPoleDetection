import torch
from torchmetrics.detection import MeanAveragePrecision
import numpy as np
import os
from glob import glob

def read_yolo_file(file_path, img_width, img_height, prediction=False):
    try:
        data = np.loadtxt(file_path)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)  # Numpy return dimensionless vector if only one line
    except:
        # File is empty, return a zero-vector
        if prediction:
            return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.int64)
        else:
            return torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64)
    
    if prediction:
        # Prediction format: class_id, score, x_center, y_center, width, height
        labels = torch.from_numpy(data[:, 0].astype(int))
        scores = torch.from_numpy(data[:, 1].astype(float))
        boxes = torch.from_numpy(data[:, 2:].astype(float))  
    else:
        # Ground truth format: class_id, x_center, y_center, width, height
        labels = torch.from_numpy(data[:, 0].astype(int))
        boxes = torch.from_numpy(data[:, 1:].astype(float))  
    
    # Validate boxes are in correct range (0-1 for normalized coordinates) (Essentially checks if the model has made a prediction outside an image)
    assert torch.all(boxes[:, 0] >= 0) and torch.all(boxes[:, 0] <= 1), "x_center out of range"
    assert torch.all(boxes[:, 1] >= 0) and torch.all(boxes[:, 1] <= 1), "y_center out of range"
    assert torch.all(boxes[:, 2] >= 0) and torch.all(boxes[:, 2] <= 1), "width out of range"
    assert torch.all(boxes[:, 3] >= 0) and torch.all(boxes[:, 3] <= 1), "height out of range"
    
    if prediction:
        return boxes, scores, labels
    else:
        return boxes, labels

def eval(pred_dir, gt_dir, img_width, img_height):
    
    metric = MeanAveragePrecision(
        box_format='xywh',  
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        rec_thresholds=None,
    )
    
    # Get all prediction files
    pred_files = sorted(glob(os.path.join(pred_dir, '*.txt')))
    
    for pred_file in pred_files:
        # Get corresponding ground truth file
        filename = os.path.basename(pred_file)
        gt_file = os.path.join(gt_dir, filename)
        

        pred_boxes, pred_scores, pred_labels = read_yolo_file(pred_file, img_width, img_height, prediction=True)
        pred_dict = {
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': pred_labels,
        }
        
        gt_boxes, gt_labels = read_yolo_file(gt_file, img_width, img_height, prediction=False)
        gt_dict = {
            'boxes': gt_boxes,
            'labels': gt_labels,
        }
        
        metric.update([pred_dict], [gt_dict])
    
    results = metric.compute()
    
    formatted_results = {
        'mAP@0.5': float(results['map_50'].numpy()),
        'mAP@0.5:0.95': float(results['map'].numpy()),
        'Precision': float(results['precision'].numpy()),
        'Recall': float(results['recall'].numpy()),
    }
    
    return formatted_results


if __name__ == "__main__":

    PRED_DIR = ""  
    GT_DIR = ""  
    
    #img_width, img_height = 1024, 128 #LiDAR
    img_width, img_height = 1920, 1208 #Real

    results = eval(PRED_DIR, GT_DIR, img_width, img_height)
    
    print("\nEvaluation Results:")
    print("--------------------------------------")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("--------------------------------------")