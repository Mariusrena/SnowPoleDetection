import time
import sys
import os 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
from torchvision.models.detection.image_list import ImageList 
import torchvision.ops as ops
import cv2

from Architectures.CNN import ConvNet
from Architectures.RPN import RPN
from Architectures.NN import ROI_NN
from Tools.rgb_dataloader import rgb_validloader
from Tools.bounding_box import draw_bb_on_prediction

nms_threshold = 0.1

with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("/home/marius/Documents/NTNU/TDT4265/SnowPoleDetection/Trained Models/CNN_RPN_ROI-NN.pt", map_location=torch.device(device))
    
    c_net = ConvNet()
    c_net.load_state_dict(model["backbone_state_dict"])
    c_net.to(device)  
    c_net.eval()

    rpn = RPN
    rpn.load_state_dict(model["rpn_state_dict"])
    rpn.to(device)
    rpn.eval()

    roi_nn = ROI_NN(6400)
    roi_nn.load_state_dict(model["roi_state_dict"])
    roi_nn.to(device)
    roi_nn.eval()

    for images, labels in rgb_validloader:

        images = images.to(device)

        features = c_net(images)
            
        image_sizes = [(img.shape[1], img.shape[2]) for img in images]
        image_list = ImageList(images, image_sizes)
            
        proposals, proposal_losses = rpn(image_list, {'0': features})    

        # Process each image in the batch separately
        for img_idx, img_proposals in enumerate(proposals):
            
            if len(img_proposals) == 0:
                continue
            
            # ROI pooling to get fixed-size features
            # Add batch dimension index for roi_align
            rois = torch.cat([
                torch.full((len(img_proposals), 1), img_idx, device=device),
                img_proposals
            ], dim=1)
            
            pooled_features = ops.roi_align(features, rois, output_size=(5, 5), spatial_scale=1.0)
            pooled_features = pooled_features.view(pooled_features.size(0), -1)
            
            # Forward pass through ROI head
            refined_bbox, class_preds = roi_nn(pooled_features)
            
            # Get class scores and predictions
            class_scores = torch.sigmoid(class_preds)

            max_scores, pred_classes = torch.max(class_scores, dim=1)
            
            # Apply confidence threshold
            confidence_keep = max_scores > 0.01
            
            # Get filtered detections
            final_boxes = refined_bbox[confidence_keep]
            final_scores = max_scores[confidence_keep]
            final_classes = pred_classes[confidence_keep]
            
            # Apply non-maximum suppression to remove overlapping detections
            nms_indices = ops.nms(final_boxes, final_scores, nms_threshold)

            draw_bb_on_prediction(images[img_idx], labels[img_idx]["boxes"].numpy(), final_boxes[nms_indices].numpy()[0])

            cv2.waitKey(0)    
            cv2.destroyAllWindows()
