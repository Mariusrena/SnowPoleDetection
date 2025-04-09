import torch
from torchmetrics.detection import MeanAveragePrecision
from Tools.rgb_dataloader import rgb_validloader

def eval(model, dataloader):
    
    metric = MeanAveragePrecision(
        box_format='xyxy',  
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        rec_thresholds=None,
    )
    
    for images, targets in dataloader:
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            pred_dict, _ = model(images, targets)

            gt_dict = targets
            
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

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = torch.load("PATH TO MODEL", device)
    model = model.to(device)
    model.eval()

    results = eval(model, rgb_validloader)
    
    print("\nEvaluation Results:")
    print("--------------------------------------")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("--------------------------------------")