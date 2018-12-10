import torch

import lib.ssds_train
from lib.utils.config_parse import cfg_from_file

cfg_from_file('./experiments/cfgs/ssd_lite_mobilenetv2_train_coco.yml')
s = lib.ssds_train.Solver()

s.test_model()