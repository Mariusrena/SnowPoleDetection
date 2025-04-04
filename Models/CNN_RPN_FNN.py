import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models.detection.image_list import ImageList 
import torchvision.ops as ops
from tqdm import tqdm, trange

import sys
import os 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Architectures.CNN import ConvNet
from Architectures.RPN import RPN
from Architectures.NN import ROI_NN
from Tools.rgb_dataloader import rgb_trainloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 20

MODEL_NAME = "CNN_RPN_ROI-NN_v2"

c_net = ConvNet().to(device)
rpn = RPN.to(device)
roi_nn = ROI_NN(6400).to(device)

optimizer = optim.Adam(list(c_net.parameters()) + list(rpn.parameters()) + list(roi_nn.parameters()), lr=0.001)

class_loss = nn.BCEWithLogitsLoss()
bbox_loss = nn.HuberLoss()

def train():
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

    for epoch in trange(NUM_EPOCHS, desc="Epochs"):
        
        epoch_loss = 0
        
        c_net.train()
        rpn.train()
        roi_nn.train()

        batch_loop = tqdm(rgb_trainloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)

        for images, labels in batch_loop:

            images = images.to(device) # Loads the image to the specified device 

            # Moves all tensor components of a structured labels list to the specified device
            labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in labels]
            
            optimizer.zero_grad() # Stes the gradient of the optimizer to zero
            
            features = c_net(images) # Passing through backbone, returns the image featuremap
            
            image_sizes = [(img.shape[1], img.shape[2]) for img in images] # Finds the size of all images in the batch (Which is always the same...)
            image_list = ImageList(images, image_sizes)
            
            proposals, proposal_losses = rpn(image_list, {'0': features}, labels) # Passing thorugh the RPN

            target_labels = [t["labels"] for t in labels] # Extracts class labels
            target_boxes = [t["boxes"] for t in labels] # Extracts bbox values

            roi_losses = 0
            total_positive = 0
            
            # Process each image in the batch separately
            for img_idx, (img_proposals, img_targets_boxes, img_targets_labels) in enumerate(zip(proposals, target_boxes, target_labels)):

                # Skip if no proposals or targets
                if len(img_proposals) == 0 or len(img_targets_boxes) == 0:
                    continue
                    
                # Calculate IoU (Intersection over Union) between proposals and target boxes
                iou_matrix = ops.box_iou(img_proposals, img_targets_boxes)
                
                # Find best ground truth box for each proposal
                max_iou, max_idx = iou_matrix.max(dim=1)

                # Select the best samples (Needs to be trained iteratively)
                pos_indices = max_iou > 0.01
                pos_count = pos_indices.sum().item()

                # Ensure minimum number of positive samples for training
                if pos_count < 16:
                    # If not enough positives, take top IoU matches
                    if len(max_iou) > 16:
                        top_indices = torch.argsort(max_iou, descending=True)[:16]
                        pos_indices = torch.zeros_like(pos_indices)
                        pos_indices[top_indices] = True
                        pos_count = 16
                
                # Skip if no positive samples
                if pos_count == 0:
                    continue
                    
                total_positive += pos_count
            
                pos_proposals = img_proposals[pos_indices] # The best proposed boxes
                pos_target_idx = max_idx[pos_indices] # Index of ground truth that is matched to a proposal
                pos_labels = img_targets_labels[pos_target_idx] # Ground truth label, always zero in this case
                pos_target_boxes = img_targets_boxes[pos_target_idx] # The targetboxed that the proposals are matched against
                
                # ROI pooling to get fixed-size features (Neural net ROI head needs this)
                # Add batch dimension index for roi_align
                rois = torch.cat([torch.full((pos_count, 1), img_idx, device=device), pos_proposals], dim=1)
                pooled_features = ops.roi_align(features, rois, output_size=(5, 5), spatial_scale=1.0)
                pooled_features = pooled_features.view(pooled_features.size(0), -1)
                
                # Forward pass through ROI head
                refined_bbox, class_preds = roi_nn(pooled_features)
                
                # Calculate classification and regression losses
                binary_labels = torch.ones_like(pos_labels, dtype=torch.float, device=device)
                roi_class_loss = class_loss(class_preds.squeeze(), binary_labels)
                roi_bbox_loss = bbox_loss(refined_bbox, pos_target_boxes)
                
                roi_losses += (roi_class_loss + roi_bbox_loss)
            
            # Calculate total loss
            if total_positive > 0:
                roi_losses = roi_losses / total_positive
                total_loss = roi_losses + sum(loss for loss in proposal_losses.values())
            else:
                # If no positive samples in the entire batch, just use RPN loss
                total_loss = sum(loss for loss in proposal_losses.values())
 
            total_loss.backward()
            optimizer.step()
            
            batch_loop.set_postfix(loss=f"{total_loss.item():.4f}")
            epoch_loss += total_loss.item()
        
        avg_loss = epoch_loss / len(rgb_trainloader)
        tqdm.write(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}')
        
        """ if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "backbone_state_dict": c_net.state_dict(),
                "rpn_state_dict": rpn.state_dict(),
                "roi_nn_state_dict": roi_nn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, f"checkpoint_epoch_{epoch+1}.pt") """



if __name__ == "__main__":
    c_net_params = sum(p.numel() for p in c_net.parameters())
    fpn_params = sum(p.numel() for p in rpn.parameters())
    roi_params = sum(p.numel() for p in roi_nn.parameters())
    
    print("#"*40)
    print(f"ConvNet Parameters: {str(c_net_params).rjust(20)}")
    print(f"FPN Parameters: {str(fpn_params).rjust(24)}")
    print(f"ROI Parameters: {str(roi_params).rjust(24)}")
    print("#"*40)
    print(f"Total parameters: {str(c_net_params + fpn_params + roi_params).rjust(22)}")
    print("#"*40)

    train()

    torch.save({
                "backbone_state_dict": c_net.state_dict(),
                "rpn_state_dict": rpn.state_dict(),
                "roi_state_dict": roi_nn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, f"{MODEL_NAME}.pt")