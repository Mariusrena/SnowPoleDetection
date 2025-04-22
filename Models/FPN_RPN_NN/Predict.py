
""" External Imports """
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import torchvision.ops as ops
from torchvision.models.detection.image_list import ImageList 
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm, trange

""" Standard Imports """
import sys
import os
import logging
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

""" Local Imports """
from FPN_RPN_NN.CNN_FPN import ConvFPN
from FPN_RPN_NN.RPN import RPN
from FPN_RPN_NN.NN import ROI_NN
from Tools.predict_dataloader import rgb_testloader, collate_fn_save

from FPN_RPN_NN.Hyperparameters import (
                                    ROI_ALIGN_OUTPUT_SIZE,
                                    SAMPLE_RATIO,
                                    NUM_CNN_OUTPUT_CHANNELS,
                                    ROI_FG_IOU_THRESH,
                                    ROI_BG_IOU_THRESH_LO, 
                                    CLASSIFICATION_LOSS_WEIGHT, 
                                    BBOX_REGRESSION_LOSS_WEIGHT, 
                                    SCORE_THRESHOLD,
                                    NMS_THRESHOLD,
                                    LR, 
                                    LR_SCHEDULER_FACTOR, 
                                    LR_SCHEDULER_PATIENCE,
                                    EARLY_STOP_PATIENCE,
                                    GRADIENT_CLIPPING_MAX_NORM,
                                    NUM_EPOCHS,
                                    RPN_ONLY,)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectDetectionModel(nn.Module):
    def __init__(self, backbone, rpn, roi_head):
        super(ObjectDetectionModel, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_head = roi_head

        self.roi_align = ops.MultiScaleRoIAlign(
            featmap_names = ["0", "1", "2"],
            output_size = (ROI_ALIGN_OUTPUT_SIZE, ROI_ALIGN_OUTPUT_SIZE),
            sampling_ratio = SAMPLE_RATIO
        )

    def forward(self, images, targets=None, score_thresh=SCORE_THRESHOLD, nms_thresh=NMS_THRESHOLD, rpn_only=RPN_ONLY):
        """
        Args:
            images (list[Tensor] or Tensor): Images to process.
            targets (list[Dict[str, Tensor]], optional): Ground truth annotations.
            score_thresh (float): Threshold to filter detections based on score during inference.
            nms_thresh (float): IoU threshold for Non-Maximum Suppression during inference.

        Returns:
            During training: (proposals, dict of losses)
            During inference: list[dict{'boxes': Tensor, 'scores': Tensor, 'labels': Tensor}]
        """

        original_image_sizes = []
        if isinstance(images, list):
            for img in images:
                val = img.shape[-2:] # Gets last to values in the image shape
                assert len(val) == 2
                original_image_sizes.append((val[0], val[1]))
            images = torch.stack(images, dim=0)
        else:
            val = images.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # Use specified backbone to get a featuremap
        features = self.backbone(images) 

        # Create ImageList (needed for RPN internal size tracking)
        # Use original_image_sizes if available, otherwise calculate from tensor
        image_list = ImageList(images, [(img.shape[-2], img.shape[-1]) for img in images])

        # Get proposals from RPN
        # Pass targets only during training

        proposals, rpn_scores, proposal_losses = self.rpn(image_list, features, targets if self.training else None)

        # INFERENCE
        if not self.training:
            if rpn_only:
                predictions = self.process_rpn_for_inference(rpn_scores[0], proposals[0], targets, image_list.image_sizes)
                return predictions
            
            # Get predictions from ROI head 
            predictions = self.process_roi_for_inference(features, proposals, image_list.image_sizes, score_thresh, nms_thresh)
            return predictions 

    @torch.no_grad()
    def process_rpn_for_inference(self, scores, proposals, targets, image_shapes, score_thresh=SCORE_THRESHOLD, nms_thresh=NMS_THRESHOLD):
        """
        Processes proposals from RPN and applies post-processing for inference.

        Args:
            scores (list[Tensor]): Scores from RPNw.
            proposals (list[Tensor]): Proposals from RPN for each image.
            image_shapes (list[tuple[int, int]]): Original image sizes (H, W).
            score_thresh (float): Score threshold for filtering.
            nms_thresh (float): NMS IoU threshold.

        Returns:
            list[dict{'boxes': Tensor, 'scores': Tensor, 'labels': Tensor}]: Final detections.
        """
        results = []
        
        # Filter by score threshold (probability of snow pole in bbox)
        keep_inds = scores >= score_thresh
        filtered_proposals = proposals[keep_inds]
        filtered_scores = scores[keep_inds]

        if keep_inds.numel() == 0:
            results.append({"boxes": torch.empty((0, 4), device=pooled_features.device),
                            "scores": torch.empty((0,), device=pooled_features.device),
                            "labels": torch.empty((0,), dtype=torch.int64, device=pooled_features.device)})

        else:
            # Clip boxes to image boundaries (likely not a problem in this task)
            img_h, img_w = image_shapes[0]
            filtered_proposals[:, [0, 2]] = filtered_proposals[:, [0, 2]].clamp(min=0, max=img_w)
            filtered_proposals[:, [1, 3]] = filtered_proposals[:, [1, 3]].clamp(min=0, max=img_h)

            # Assign labels - Binary classification (label 1 for "object")
            filtered_labels = torch.ones_like(filtered_scores, dtype=torch.int64)

            # Apply Non-Maximum Suppression (NMS)
            nms_keep_indices = ops.nms(filtered_proposals, filtered_scores, nms_thresh)

            final_boxes = filtered_proposals[nms_keep_indices]
            final_scores = filtered_scores[nms_keep_indices]
            final_labels = filtered_labels[nms_keep_indices] # All will be 1

            results.append({"boxes": final_boxes, "scores": final_scores, "labels": final_labels})

        return results

    @torch.no_grad()
    def process_roi_for_inference(self, features, proposals, image_shapes, score_thresh, nms_thresh):
        """
        Processes proposals through ROI head and applies post-processing for inference.

        Args:
            features (Tensor or Dict[str, Tensor]): Feature maps from backbone.
            proposals (list[Tensor]): Proposals from RPN for each image.
            image_shapes (list[tuple[int, int]]): Original image sizes (H, W).
            score_thresh (float): Score threshold for filtering.
            nms_thresh (float): NMS IoU threshold.

        Returns:
            list[dict{'boxes': Tensor, 'scores': Tensor, 'labels': Tensor}]: Final detections.
        """
        results = []
        batch_size = len(proposals)
        for img_idx in range(batch_size):
            img_proposals = proposals[img_idx]

            if len(img_proposals) == 0: continue

            rois = torch.cat([
                torch.full((len(img_proposals), 1), img_idx, device=img_proposals.device),
                img_proposals
            ], dim=1)

            # The feature maps from the backbone is aligened with the RPN map to use in the ROI head
            pooled_features = self.roi_align(features,                
                                            [img_proposals],          
                                            [image_shapes[img_idx]]
                                        )
            pooled_features = pooled_features.view(pooled_features.size(0), -1)

            box_refinement, class_scores = self.roi_head(pooled_features)

            # Convert logits to probabilities using sigmoid
            scores = torch.sigmoid(class_scores.squeeze(-1)) # Remove last dim if it's 1
            # Filter by score threshold (probability of snow pole in bbox)
            keep_inds = scores >= score_thresh
            filtered_proposals = img_proposals[keep_inds]
            filtered_box_refinement = box_refinement[keep_inds]
            filtered_scores = scores[keep_inds]

            if keep_inds.numel() == 0:
                results.append({"boxes": torch.empty((0, 4), device=pooled_features.device),
                                "scores": torch.empty((0,), device=pooled_features.device),
                                "labels": torch.empty((0,), dtype=torch.int64, device=pooled_features.device)})
                continue

            # Applying the inverse transformation to get final boxes
            proposal_widths = filtered_proposals[:, 2] - filtered_proposals[:, 0]
            proposal_heights = filtered_proposals[:, 3] - filtered_proposals[:, 1]
            proposal_x_centers = (filtered_proposals[:, 0] + filtered_proposals[:, 2]) / 2
            proposal_y_centers = (filtered_proposals[:, 1] + filtered_proposals[:, 3]) / 2

            predicted_tx = filtered_box_refinement[:, 0]
            predicted_ty = filtered_box_refinement[:, 1]
            predicted_tw = filtered_box_refinement[:, 2]
            predicted_th = filtered_box_refinement[:, 3]

            pred_x_centers = proposal_x_centers + predicted_tx * proposal_widths
            pred_y_centers = proposal_y_centers + predicted_ty * proposal_heights
            pred_widths = proposal_widths * torch.exp(predicted_tw)
            pred_heights = proposal_heights * torch.exp(predicted_th)

            # Convert back to (xmin, ymin, xmax, ymax)
            filtered_boxes = torch.zeros_like(filtered_proposals)
            filtered_boxes[:, 0] = pred_x_centers - pred_widths / 2
            filtered_boxes[:, 1] = pred_y_centers - pred_heights / 2
            filtered_boxes[:, 2] = pred_x_centers + pred_widths / 2
            filtered_boxes[:, 3] = pred_y_centers + pred_heights / 2

            # Clip boxes to image boundaries (likely not a problem in this task)
            img_h, img_w = image_shapes[img_idx]
            filtered_boxes[:, [0, 2]] = filtered_boxes[:, [0, 2]].clamp(min=0, max=img_w)
            filtered_boxes[:, [1, 3]] = filtered_boxes[:, [1, 3]].clamp(min=0, max=img_h)

            # Assign labels - Binary classification (label 1 for "object")
            filtered_labels = torch.ones_like(filtered_scores, dtype=torch.int64)

            # Apply Non-Maximum Suppression (NMS)
            nms_keep_indices = ops.nms(filtered_boxes, filtered_scores, nms_thresh)

            final_boxes = filtered_boxes[nms_keep_indices]
            final_scores = filtered_scores[nms_keep_indices]
            final_labels = filtered_labels[nms_keep_indices] # All will be 1

            results.append({"boxes": final_boxes, "scores": final_scores, "labels": final_labels})

        return results

def validate_model(model, val_loader, device):
    model.eval()
    
    with torch.no_grad():
        metric = MeanAveragePrecision(
            iou_type="bbox",
            box_format='xyxy',
        )
        
        for images, targets in val_loader:
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            pred_dict = model(images)

            metric.update(pred_dict, targets)

        results = metric.compute()

        formatted_results = {
            'mAP@0.5': float(results['map_50'].numpy()) if float(results['map_50'].numpy()) > 0 else 0,
            'mAP@0.5:0.95': float(results['map'].numpy()) if float(results['map'].numpy()) > 0 else 0,
            'mAR@1': float(results['mar_1'].numpy()) if float(results['mar_1'].numpy()) > 0 else 0,
            'mAR@10': float(results['mar_10'].numpy()) if float(results['mar_10'].numpy()) > 0 else 0,
            'mAR@100': float(results['mar_100'].numpy()) if float(results['mar_100'].numpy()) > 0 else 0,
        }
    
        logger.info(f"Evaluation Results:")
        logger.info("--------------------------------------")
        for metric, value in formatted_results.items():
            logger.info(f"{metric}: {value:.4f}")
            logger.info("--------------------------------------")
        #return formatted_results

def save_predictions(model, data_loader, device, output_dir, score_threshold=SCORE_THRESHOLD):
    """
    Runs inference on the data_loader and saves predictions to text files.

    Args:
        model (nn.Module): The trained object detection model.
        data_loader (DataLoader): DataLoader providing images, targets, and filenames.
        device (torch.device): The device to run inference on (e.g., 'cuda').
        output_dir (str): Directory to save the prediction .txt files.
        score_threshold (float): Minimum score to keep a detection.
    """
    model.eval()  # Set model to evaluation mode
    os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist
    logger.info(f"Saving predictions to: {output_dir}")

    with torch.no_grad(): # Disable gradient calculations
        for batch in tqdm(data_loader, desc="Saving Predictions"):
            if batch is None or len(batch) != 3:
                logger.warning("Skipping empty or malformed batch.")
                continue

            images, _, filenames = batch # We don't need targets for inference

            if not images or not filenames:
                logger.warning("Skipping batch with missing images or filenames.")
                continue

            # Assuming batch_size=1 due to collate_fn_save and dataloader setup
            if len(images) != 1 or len(filenames) != 1:
                 logger.error(f"Expected batch size 1, but got {len(images)}. Skipping batch.")
                 continue

            image_tensor = images[0].to(device) # Get the single image tensor, move to device
            current_filename = filenames[0]

            # The model expects a list of tensors, even for a single image
            pred_dict_list = model([image_tensor]) # Pass image tensor within a list

            # Since batch_size=1, pred_dict_list contains results for one image
            if not pred_dict_list:
                logger.warning(f"Model returned no predictions for {current_filename}.")
                continue

            predictions = pred_dict_list[0] # Get the dictionary for the first (only) image

            # Get image dimensions (height, width) from the tensor passed to the model
            # This assumes the tensor shape is (C, H, W)
            img_h, img_w = image_tensor.shape[-2:]

            output_filename = os.path.splitext(current_filename)[0] + ".txt"
            output_filepath = os.path.join(output_dir, output_filename)

            with open(output_filepath, "w") as f:
                boxes = predictions["boxes"] # Shape: [N, 4] in xyxy format
                scores = predictions["scores"] # Shape: [N]
                labels = predictions["labels"] # Shape: [N]

                for i in range(boxes.shape[0]):
                    score = scores[i].item()
                    if score < score_threshold:
                        continue # Skip detections below threshold

                    box = boxes[i].cpu().numpy() # xyxy format
                    xmin, ymin, xmax, ymax = box

                    # Convert xyxy to normalized xywh format
                    box_width = xmax - xmin
                    box_height = ymax - ymin
                    x_center = (xmin + xmax) / 2
                    y_center = (ymin + ymax) / 2

                    # Normalize coordinates
                    norm_x_center = x_center / img_w
                    norm_y_center = y_center / img_h
                    norm_width = box_width / img_w
                    norm_height = box_height / img_h

                    # Clamp normalized values to [0, 1] to avoid floating point issues
                    norm_x_center = max(0.0, min(1.0, norm_x_center))
                    norm_y_center = max(0.0, min(1.0, norm_y_center))
                    norm_width = max(0.0, min(1.0, norm_width))
                    norm_height = max(0.0, min(1.0, norm_height))


                    # *** CLASS ID ***
                    # Your requested format starts with 0. Your model predicts label 1.
                    # Using 0 here based on your request. Verify this is correct.
                    class_id = 0
                    # If you want the model's predicted label (which is 1 in your setup):
                    # class_id = labels[i].item()

                    # Format: class x_center y_center width height score
                    f.write(f"{class_id} {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f} {score:.6f}\n")

    logger.info("Finished saving predictions.")


if __name__ == "__main__":
    backbone = ConvFPN()
    rpn = RPN
    roi_head = ROI_NN(int(NUM_CNN_OUTPUT_CHANNELS*ROI_ALIGN_OUTPUT_SIZE**2)) # Pass number of input features

    # Create integrated model
    model = ObjectDetectionModel(backbone, rpn, roi_head)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model 
    model_params = torch.load("/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Trained_Models/15M/Current_best_model.pt", weights_only=True, map_location=device)
    model.load_state_dict(model_params["model_state_dict"])

    
    model = model.to(device)

    prediction_output_dir = "/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Trained_Models/15M/Test_Predictions"

    # --- Run Prediction Saving ---
    save_predictions(model, rgb_testloader, device, prediction_output_dir, score_threshold=SCORE_THRESHOLD) # Use the prediction loader
