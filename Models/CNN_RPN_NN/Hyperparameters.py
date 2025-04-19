""" 
    Hyperparameters present in the ObjectDetection.py script
    Hyperparameters for CNN, RPN and ROI Head is found inside respective submodules
"""

# ROI Align
ROI_ALIGN_OUTPUT_SIZE = 7 # e.g. 3x3
ROI_ALIGN_SPATIAL_SCALE = 1.0 / (2**3) # Calculated based on ConvNet stride (5 maxpools)

# CNN
NUM_CNN_OUTPUT_CHANNELS = 512

# Training
ROI_FG_IOU_THRESH = 0.5
ROI_BG_IOU_THRESH_LO = 0.3
CLASSIFICATION_LOSS_WEIGHT = 1.0 
BBOX_REGRESSION_LOSS_WEIGHT = 1.0 

# Inference
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

# Training Loop
LR = 0.000001
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 100
EARLY_STOP_PATIENCE = 100
GRADIENT_CLIPPING_MAX_NORM = 10.0
NUM_EPOCHS = 2000
