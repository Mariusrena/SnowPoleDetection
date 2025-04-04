import torch
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.image_list import ImageList 

anchor_generator = AnchorGenerator(
    sizes=((8, 16, 32, 64, 128, 256),),  
    aspect_ratios=((0.25, 0.5, 2, 4),)
)

#Set the number of channels from backbone.
in_channels = 256

# Determine the number of anchors per spatial location.
num_anchors = anchor_generator.num_anchors_per_location()[0]

# Create the RPN head, used to predict bounding boxes
rpn_head = RPNHead(in_channels, num_anchors)

# Instantiate the RegionProposalNetwork with desired parameters.
RPN = RegionProposalNetwork(
    anchor_generator,
    rpn_head,
    fg_iou_thresh=0.7,             # IoU threshold for an object anchor
    bg_iou_thresh=0.3,             # IoU threshold for a background anchor.
    batch_size_per_image=256,      # Total anchors sampled per image during training.
    positive_fraction=0.5,         # Fraction of positive anchors in the mini-batch.
    pre_nms_top_n=dict(training=2000, testing=200),  # Number of proposals to consider before NMS.
    post_nms_top_n=dict(training=1000, testing=500), # Number of proposals to keep after NMS.
    nms_thresh=0.1                 # NMS threshold.
)
