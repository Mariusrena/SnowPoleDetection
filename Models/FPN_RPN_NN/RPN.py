from typing import Dict, List, Optional, Tuple
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork, concat_box_prediction_layers
from torch import Tensor
import torch.nn as nn
from torchvision.models.detection.image_list import ImageList 

""" Standard Imports """
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from FPN_RPN_NN.Hyperparameters import NUM_CNN_OUTPUT_CHANNELS

""" 
    Based on parameters from dataset:
        Min ratio: 0.016903832084823132 (Width/height)
        Max ratio: 0.24063788540833286 (Width/height)
        Min bbox size: 350 (pixels)
        Max bbox size: 60490 (pixels)
"""

anchor_sizes = ((18, 38, 60, 120, 256), (18, 38, 60, 120, 256), (18, 38, 60, 120, 256),)
aspect_ratios = ((5.23, 8.59, 12.06, 16.75, 26.54),) * len(anchor_sizes) 

anchor_generator = AnchorGenerator(
    sizes=anchor_sizes,
    aspect_ratios=aspect_ratios
)

class CutsomRPN(RegionProposalNetwork):
    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, scores, losses


rpn_head = RPNHead(
    NUM_CNN_OUTPUT_CHANNELS,
    anchor_generator.num_anchors_per_location()[0],
    conv_depth = 5
)

# Instantiate the RegionProposalNetwork with desired parameters.
RPN = CutsomRPN(
    anchor_generator,
    rpn_head,
    fg_iou_thresh = 0.7,                                  # IoU threshold for an object anchor
    bg_iou_thresh = 0.3,                                  # IoU threshold for a background anchor.
    batch_size_per_image = 256,                           # Total anchors sampled per image during training.
    positive_fraction = 0.5,                              # Fraction of positive anchors in the mini-batch.
    pre_nms_top_n = dict(training=2000, testing=1000),    # Number of proposals to consider before NMS.
    post_nms_top_n = dict(training=1000, testing=500),    # Number of proposals to keep after NMS.
    nms_thresh = 0.7,                                     # NMS threshold
)

