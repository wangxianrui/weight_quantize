# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")
COCO_API = os.path.join(HOME, "coco/PythonAPI")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

###############################
# select the dataset COCO
###############################
num_classes = 81
epochs = 100
lr_steps = (280000, 360000, 400000)
max_iter = 400000
feature_maps = [50, 50, 25, 13, 11, 9]
min_dim = 800
steps = [16, 16, 32, 64, 73, 89]
min_sizes = [21, 45, 99, 153, 207, 261]
max_sizes = [45, 99, 153, 207, 261, 315]
aspect_ratios = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
variance = [0.1, 0.2]
clip = True
name = 'COCO'
data_root = "/home/wxrui/data/coco"

train = {
    'milestones': [40, 80, 90, 100],
    'dataset': 'COCO',  # VOC or COCO
    'dataset_root': data_root,  # Dataset root directory path
    'basenet': '',  # Pretrained base model
    'batch_size': 8,  # Batch size for training
    'resume': '',  # Checkpoint state_dict file to resume training from
    'start_iter': 0,  # Resume training at this iter
    'num_workers': 4,  # Number of workers used in dataloading
    'cuda': True,  # Use CUDA to train model
    'lr': 1e-3,  # initial learning rate
    'momentum': 0.9,  # Momentum value for optim
    'weight_decay': 5e-4,  # Weight decay for SGD
    'gamma': 0.1,  # Gamma update for SGD
    'visdom': False,  # Use visdom for loss visualization
    'save_folder': 'weights/',  # Directory for saving checkpoint models
}

test = {
    'dataset': 'COCO',
    'trained_model': 'models/peleenet_ssd/weights/COCO30.6.pth',  # Trained state_dict file path to open
    'save_folder': 'eval/',  # Dir to save results
    'visual_threshold': 0.6,  # Final confidence threshold
    'cuda': True,  # Use cuda to train model
    'retest': False,
    'dataset_root': data_root,  # Location of VOC root directory
}

'''
###############################
# select the dataset VOC
###############################
num_classes = 21
# epochs = 100
# lr_steps = (80000, 100000, 120000)
# max_iter = 120000
feature_maps = [50, 50, 25, 13, 11, 9]
min_dim = 800
steps = [16, 16, 32, 64, 73, 89]
min_sizes = [30, 60, 111, 162, 213, 264]
max_sizes = [60, 111, 162, 213, 264, 315]
aspect_ratios = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
variance = [0.1, 0.2]
clip = True
name = 'VOC'
data_root = "/home/wxrui/DATA/VOCdevkit/"

train = {
    'milestones': [40, 80, 90, 100],
    'dataset': 'VOC',  # VOC or COCO
    'dataset_root': data_root,  # Dataset root directory path
    'basenet': '',  # Pretrained base model
    'batch_size': 16,  # Batch size for training
    'resume': '',  # Checkpoint state_dict file to resume training from
    'start_iter': 0,  # Resume training at this iter
    'num_workers': 4,  # Number of workers used in dataloading
    'cuda': True,  # Use CUDA to train model
    'lr': 1e-3,  # initial learning rate
    'momentum': 0.9,  # Momentum value for optim
    'weight_decay': 5e-4,  # Weight decay for SGD
    'gamma': 0.1,  # Gamma update for SGD
    'visdom': False,  # Use visdom for loss visualization
    'save_folder': 'weights/',  # Directory for saving checkpoint models
}

test = {
    'dataset': 'VOC',  # VOC or COCO
    'trained_model': 'weights/VOC4.pth',  # Trained state_dict file path to open
    'save_folder': 'eval/',  # Dir to save results
    'visual_threshold': 0.6,  # Final confidence threshold
    'cuda': True,  # Use cuda to train model
    'retest': False,
    'dataset_root': data_root,  # Location of VOC root directory
}
'''
