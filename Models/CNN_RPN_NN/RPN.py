from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork

""" Standard Imports """
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Models.Hyperparameters import NUM_CNN_OUTPUT_CHANNELS

anchor_generator = AnchorGenerator(
    sizes=((16, 64, 96, 128, 256),),  
    #aspect_ratios=((0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1, 2),)
    aspect_ratios=((0.1, 0.25, 0.5, 1, 3, 5, 10),)
)

#Set the number of channels from backbone.
in_channels = NUM_CNN_OUTPUT_CHANNELS

# Determine the number of anchors per spatial location.
num_anchors = anchor_generator.num_anchors_per_location()[0]

# Create the RPN head, used to predict bounding boxes
rpn_head = RPNHead(in_channels, num_anchors)

# Instantiate the RegionProposalNetwork with desired parameters.
RPN = RegionProposalNetwork(
    anchor_generator,
    rpn_head,
    fg_iou_thresh = 0.5,                                  # IoU threshold for an object anchor
    bg_iou_thresh = 0.3,                                  # IoU threshold for a background anchor.
    batch_size_per_image = 2048,                          # Total anchors sampled per image during training.
    positive_fraction = 0.5,                             # Fraction of positive anchors in the mini-batch.
    pre_nms_top_n = dict(training=2000, testing=1000),    # Number of proposals to consider before NMS.
    post_nms_top_n = dict(training=1000, testing=500),    # Number of proposals to keep after NMS.
    nms_thresh = 0.5,                                     # NMS threshold
    #score_thresh = 0.5
)