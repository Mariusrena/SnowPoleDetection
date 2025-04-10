import torch
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.image_list import ImageList 

anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),  
    aspect_ratios=((0.1 ,0.25, 0.5, 1))
)

#Set the number of channels from backbone.
in_channels = 512

# Determine the number of anchors per spatial location.
num_anchors = anchor_generator.num_anchors_per_location()[0]

# Create the RPN head, used to predict bounding boxes
rpn_head = RPNHead(in_channels, num_anchors)

# Instantiate the RegionProposalNetwork with desired parameters.
RPN = RegionProposalNetwork(
    anchor_generator,
    rpn_head,
    fg_iou_thresh=0.5,             # IoU threshold for an object anchor
    bg_iou_thresh=0.1,             # IoU threshold for a background anchor.
    batch_size_per_image=1024,      # Total anchors sampled per image during training.
    positive_fraction=0.9,         # Fraction of positive anchors in the mini-batch.
    pre_nms_top_n=dict(training=2000, testing=1000),  # Number of proposals to consider before NMS.
    post_nms_top_n=dict(training=1000, testing=300), # Number of proposals to keep after NMS.
    nms_thresh=0.3                # NMS threshold.
)