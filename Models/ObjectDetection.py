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

""" Local Imports """
from Architectures.CNN import ConvNet
from Architectures.RPN import RPN
from Architectures.NN import ROI_NN
from Tools.rgb_dataloader import rgb_trainloader, rgb_validloader

from Models.Hyperparameters import (ROI_FG_IOU_THRESH, 
                                    ROI_BG_IOU_THRESH_LO, 
                                    CLASSIFICATION_LOSS_WEIGHT, 
                                    BBOX_REGRESSION_LOSS_WEIGHT, 
                                    LR, 
                                    LR_SCHEDULER_FACTOR, 
                                    LR_SCHEDULER_PATIENCE,
                                    EARLY_STOP_PATIENCE,
                                    SCORE_THRESHOLD,
                                    NMS_THRESHOLD)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectDetectionModel(nn.Module):
    def __init__(self, backbone, rpn, roi_head):
        super(ObjectDetectionModel, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_head = roi_head

    def forward(self, images, targets=None, score_thresh=SCORE_THRESHOLD, nms_thresh=NMS_THRESHOLD):
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
        proposals, proposal_losses = self.rpn(image_list, {'0': features}, targets if self.training else None)

        # INFERENCE
        if not self.training:
            # Get predictions from ROI head 
            predictions = self.process_roi_for_inference(features, proposals, image_list.image_sizes, score_thresh, nms_thresh)
            return predictions 
        
        # TRAINING
        else:
            if targets is None:
                raise ValueError("Targets should not be None during training")

            # Process proposals with ROI head for loss calculation
            roi_losses = self.process_roi_proposals(features, proposals, targets)

            # Combine losses from RPN and ROI Head
            total_losses = {}
            total_losses.update(proposal_losses)
            if roi_losses is not None:
                # Ensure the key matches what train_model expects
                total_losses['loss_roi'] = roi_losses

            return total_losses


    def process_roi_proposals(self, features, proposals, targets):
        """
        Processes proposals through ROI head and calculated loss for training.

        Args:
            features (Tensor or Dict[str, Tensor]): Feature maps from backbone.
            proposals (list[Tensor]): Proposals from RPN for each image.
            targets (list[Tensor]): Targets from dataset for each image.

        Returns:
            float: Calculated Loss.
        """
        spatial_scale = 1.0 / (2**5) # Calculated based on ConvNet stride (5 maxpools)
        batch_size = len(proposals)
        roi_losses = 0
        valid_batches = 0 # Count images that contribute to loss

        for img_idx in range(batch_size):
            img_proposals = proposals[img_idx]
            target_boxes = targets[img_idx]["boxes"]

            if len(img_proposals) == 0: continue
            if len(target_boxes) == 0: continue

            iou_matrix = ops.box_iou(img_proposals, target_boxes)

            max_iou, matched_gt_idx = iou_matrix.max(dim=1) # Finds the target that best matches a proposal

            # Assigns -1 to all proposals
            assigned_labels = torch.full((len(img_proposals),), -1, dtype=torch.float32, device=img_proposals.device)
            pos_indices = torch.where(max_iou >= ROI_FG_IOU_THRESH)[0]
            assigned_labels[pos_indices] = 1.0 # Reassigns to 1 if object
            neg_indices = torch.where(max_iou < ROI_BG_IOU_THRESH_LO)[0]
            assigned_labels[neg_indices] = 0.0 # Reassigns to 0 if not an object

            # Proposals in between threshold stay -1 => irrelevant
            valid_indices = torch.where(assigned_labels >= 0)[0]
            if len(valid_indices) == 0: continue

            rois = torch.cat([
                torch.full((len(img_proposals), 1), img_idx, device=img_proposals.device),
                img_proposals
            ], dim=1)

            # Use feature map corresponding to the batch index if features is a dict/list
            # The feature map from the backbone is aligened with the RPN map to use in the ROI head
            current_features = features['0'] if isinstance(features, dict) else features
            pooled_features = ops.roi_align(
                current_features, rois, output_size=(3, 3), spatial_scale=spatial_scale
            )
            pooled_features = pooled_features.view(pooled_features.size(0), -1)

            # Box refinement, given as the parameterized offsets of the proposals from targets
            # Probability that a bbox includes a Snpwpole
            box_refinement, class_scores = self.roi_head(pooled_features)

            # Only use fg/bg samples, not the ones in between
            sampled_assigned_labels = assigned_labels[valid_indices]
            sampled_class_scores = class_scores[valid_indices].squeeze(-1)

            # Classification
            # With logits as autocast is unhappy with "normal" BCE
            roi_class_loss = F.binary_cross_entropy_with_logits(
                sampled_class_scores, sampled_assigned_labels
            )

            # Get the indices of the positive samples within the sampled_assigned_labels
            pos_mask_in_sampled = (sampled_assigned_labels == 1)
            pos_indices_in_original = valid_indices[pos_mask_in_sampled]

            # BBox regression
            if pos_indices_in_original.numel() > 0:
                # Get the indices of the positive samples within the *valid* indices
                pos_indices_in_valid = torch.where(sampled_assigned_labels == 1)[0]

                # Use indices to select the corresponding box refinements
                pos_box_refinement = box_refinement[pos_indices_in_valid]
                pos_proposals = img_proposals[pos_indices_in_original]
                gt_boxes = target_boxes[matched_gt_idx[pos_indices_in_original]]

                # Calculate parameterized regression targets
                proposal_widths = pos_proposals[:, 2] - pos_proposals[:, 0]
                proposal_heights = pos_proposals[:, 3] - pos_proposals[:, 1]
                proposal_x_centers = (pos_proposals[:, 0] + pos_proposals[:, 2]) / 2
                proposal_y_centers = (pos_proposals[:, 1] + pos_proposals[:, 3]) / 2

                gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
                gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
                gt_x_centers = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
                gt_y_centers = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

                tx = (gt_x_centers - proposal_x_centers) / proposal_widths
                ty = (gt_y_centers - proposal_y_centers) / proposal_heights
                tw = torch.log(gt_widths / proposal_widths)
                th = torch.log(gt_heights / proposal_heights)

                regression_targets = torch.stack((tx, ty, tw, th), dim=1)

                roi_bbox_loss = F.smooth_l1_loss(
                    pos_box_refinement, regression_targets, beta=1.0
                )
            else:
                roi_bbox_loss = torch.tensor(1000, device=current_features.device)

            # Combined loss, might weighting
            img_loss = CLASSIFICATION_LOSS_WEIGHT * roi_class_loss + BBOX_REGRESSION_LOSS_WEIGHT * roi_bbox_loss

            if torch.isnan(img_loss) or torch.isinf(img_loss): continue

            roi_losses += img_loss
            valid_batches += 1

        average_roi_loss = roi_losses / max(valid_batches, 1)
        return average_roi_loss

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
        spatial_scale = 1.0 / (2**5) # Calculated based on ConvNet stride (5 maxpools)
        batch_size = len(proposals)
        for img_idx in range(batch_size):
            img_proposals = proposals[img_idx]

            if len(img_proposals) == 0: continue

            rois = torch.cat([
                torch.full((len(img_proposals), 1), img_idx, device=img_proposals.device),
                img_proposals
            ], dim=1)

            # Use feature map corresponding to the batch index if features is a dict/list
            # The feature map from the backbone is aligened with the RPN map to use in the ROI head
            current_features = features['0'] if isinstance(features, dict) else features
            pooled_features = ops.roi_align(
                current_features, rois, output_size=(3, 3), spatial_scale=spatial_scale
            )
            pooled_features = pooled_features.view(pooled_features.size(0), -1)

            box_refinement, class_scores = self.roi_head(pooled_features)

            # Convert logits to probabilities using sigmoid
            scores = torch.sigmoid(class_scores.squeeze(-1)) # Remove last dim if it's 1

            # Filter by score threshold
            keep_inds = scores >= score_thresh
            filtered_proposals = img_proposals[keep_inds]
            filtered_box_refinement = box_refinement[keep_inds]
            filtered_scores = scores[keep_inds]

            if keep_inds.numel() == 0:
                results.append({"boxes": torch.empty((0, 4), device=current_features.device),
                                "scores": torch.empty((0,), device=current_features.device),
                                "labels": torch.empty((0,), dtype=torch.int64, device=current_features.device)})
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

            # Clip boxes to image boundaries
            img_h, img_w = image_shapes[img_idx]
            filtered_boxes[:, [0, 2]] = filtered_boxes[:, [0, 2]].clamp(min=0, max=img_w)
            filtered_boxes[:, [1, 3]] = filtered_boxes[:, [1, 3]].clamp(min=0, max=img_h)

            # Assign labels - Binary classification (label 0 for "object")
            filtered_labels = torch.zeros_like(filtered_scores, dtype=torch.int64)

            # Apply Non-Maximum Suppression (NMS)
            nms_keep_indices = ops.nms(filtered_boxes, filtered_scores, nms_thresh)

            final_boxes = filtered_boxes[nms_keep_indices]
            final_scores = filtered_scores[nms_keep_indices]
            final_labels = filtered_labels[nms_keep_indices] # All will be 0

            results.append({"boxes": final_boxes, "scores": final_scores, "labels": final_labels})

        return results


def train_model(model, train_loader, val_loader=None, num_epochs=100, device="cuda"):
    # Device configuration
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)
    
    # Gradient scaler for mixed precision training
    scaler = GradScaler(device)
    
    # Early stopping variables
    best_val_result = float('inf')
    early_stop_counter = 0
    
    # Training loop
    for epoch in trange(num_epochs, desc="Epochs"):
        
        model.train()
        train_loss = 0.0
        batch_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for images, targets in batch_loop:
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass with mixed precision
                with autocast(str(device)):
                    losses = model(images, targets)
                    loss = sum(loss for loss in losses.values())
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                # Update progress
                batch_loss = loss.item()
                train_loss += batch_loss
                batch_loop.set_postfix(loss=f"{batch_loss:.4f}")
            
            except Exception as e:
                logger.error(f"Error in training batch: {e}")
                continue
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_train_loss)

        # Validation phase
        if val_loader is not None:
            val_results = validate_model(model, val_loader, device)

            logger.info(f"Evaluation Results @ Epoch {epoch+1}/{num_epochs}:")
            logger.info("--------------------------------------")
            for metric, value in val_results.items():
                logger.info(f"{metric}: {value:.4f}")
            logger.info("--------------------------------------")
            
            # Early stopping check
            if val_results['mAP@0.5'] < best_val_result:
                best_val_result = val_results['mAP@0.5']
                early_stop_counter = 0
                
                # Save best model
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": best_val_result,
                }, "best_model.pt")
                
            else:
                early_stop_counter += 1
                if early_stop_counter >= EARLY_STOP_PATIENCE and val_results['mAP@0.5'] < 9998:
                    logger.info(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        if (epoch + 1) % 20 == 0:
            torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": best_val_result,
                }, f"checkpoint_{epoch+1}.pt")

    
    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, "final_model.pt")
    
    return model


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
            'mAP@0.5': float(results['map_50'].numpy()) if float(results['map_50'].numpy()) > 0 else 9999,
            'mAP@0.5:0.95': float(results['map'].numpy()) if float(results['map'].numpy()) > 0 else 9999,
            'mAR@1': float(results['mar_1'].numpy()) if float(results['mar_1'].numpy()) > 0 else 9999,
            'mAR@10': float(results['mar_10'].numpy()) if float(results['mar_10'].numpy()) > 0 else 9999,
            'mAR@100': float(results['mar_100'].numpy()) if float(results['mar_100'].numpy()) > 0 else 9999,
        }
    
        return formatted_results

if __name__ == "__main__":
    # Model components
    backbone = ConvNet()
    rpn = RPN
    roi_head = ROI_NN(4608)
    
    # Create integrated model
    model = ObjectDetectionModel(backbone, rpn, roi_head)

    # Load model 
    # model_params = torch.load("/home/mariumre/Documents/SnowPoleDetection/Models/current_best.pt", weights_only=True)
    # model.load_state_dict(model_params["model_state_dict"])

    # Print model parameters
    backbone_params = sum(p.numel() for p in backbone.parameters())
    rpn_params = sum(p.numel() for p in rpn.parameters())
    roi_params = sum(p.numel() for p in roi_head.parameters())
    
    print("#"*40)
    print(f"Backbone Parameters: {str(backbone_params).rjust(19)}")
    print(f"RPN Parameters: {str(rpn_params).rjust(24)}")
    print(f"ROI Parameters: {str(roi_params).rjust(24)}")
    print("#"*40)
    print(f"Total parameters: {str(backbone_params + rpn_params + roi_params).rjust(22)}")
    print("#"*40)
    
    # Train model
    train_model(model, rgb_trainloader, rgb_validloader, num_epochs=2000)