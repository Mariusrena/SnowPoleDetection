from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
import torch.nn as nn

""" Standard Imports """
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from FPN_RPN_NN.Hyperparameters import NUM_CNN_OUTPUT_CHANNELS

""" 
    Based on parameters from dataset:
        Min ratio: 0.016903832084823132
        Max ratio: 0.24063788540833286
        Min bbox size: 350 (pixels)
        Max bbox size: 60490 (pixels)
"""

anchor_sizes = ((32, 64), (64, 128), (128, 256), )
aspect_ratios = ((0.25, 0.1, 0.05),) * len(anchor_sizes) 

anchor_generator = AnchorGenerator(
    sizes=anchor_sizes,
    aspect_ratios=aspect_ratios
)


rpn_head = RPNHead(
    NUM_CNN_OUTPUT_CHANNELS,
    anchor_generator.num_anchors_per_location()[0]
)

# Instantiate the RegionProposalNetwork with desired parameters.
RPN = RegionProposalNetwork(
    anchor_generator,
    rpn_head,
    fg_iou_thresh = 0.5,                                  # IoU threshold for an object anchor
    bg_iou_thresh = 0.1,                                  # IoU threshold for a background anchor.
    batch_size_per_image = 1024,                          # Total anchors sampled per image during training.
    positive_fraction = 0.5,                             # Fraction of positive anchors in the mini-batch.
    pre_nms_top_n = dict(training=2000, testing=1000),    # Number of proposals to consider before NMS.
    post_nms_top_n = dict(training=1000, testing=500),    # Number of proposals to keep after NMS.
    nms_thresh = 0.7,                                     # NMS threshold
)