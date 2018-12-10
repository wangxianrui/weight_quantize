import os
import argparse
import torch
from .ssd import build_ssd
from .data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from .eval import test_net


class args:
    num_classes = 21
    trained_model = 'weights/ssd300_mAP_77.43_v2.pth'
    voc_root = '/home/wxrui/DATA/VOCdevkit/'
    cuda = True


net = build_ssd('test', 300, args.num_classes)  # initialize SSD
net.load_state_dict(torch.load(args.trained_model))
net.eval()

# load data
dataset_mean = (104, 117, 123)
dataset = VOCDetection(args.voc_root, [('2007', 'test')],
                       BaseTransform(300, dataset_mean),
                       VOCAnnotationTransform())
if args.cuda:
    net = net.cuda()

# evaluation
test_net(net, dataset)
