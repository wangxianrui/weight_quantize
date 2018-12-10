from __future__ import print_function
import sys

import pickle
import time

import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from .data_test import AnnotationTransform, COCODetection, VOCDetection, BaseTransform, preproc
import torch.utils.data as data
from .peleenet_ssd import build_ssd
from .utils.nms_wrapper import nms
from .utils.timer import Timer
from .layers import *


def test(config, model):
    if config.test['cuda'] and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(config.test['save_folder']):
        os.mkdir(config.test['save_folder'])
    test_save_dir = os.path.join(config.test['save_folder'], 'ss_predict')
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)

    num_classes = config.num_classes

    net = model
    # net.load_state_dict(torch.load(config.test['trained_model']))

    if config.test['cuda']:
        net = torch.nn.DataParallel(model.cuda())
        cudnn.benchmark = True

    if config.test['cuda']:
        net.cuda()
        cudnn.benchmark = True
    # load dataset
    print('Loading Dataset...')
    if config.test['dataset'] == 'VOC':
        testset = VOCDetection(
            config.test['dataset_root'], [('2007', 'test')], None, AnnotationTransform())
    elif config.test['dataset'] == 'COCO':
        testset = COCODetection(
            config.test['dataset_root'], [('2014', 'minival')], None)
    else:
        print('Only VOC and COCO are supported now!')
        exit()

    net.eval()
    print('Finished loading model!')
    top_k = (300, 200)[config.test['dataset'] == 'COCO']
    rgb_std = (1, 1, 1)
    if config.test['dataset'] == 'VOC':
        APs, mAP = test_net(config, test_save_dir, net, config.test['cuda'], testset, num_classes,
                            BaseTransform(config.min_dim, config.MEANS, rgb_std, (2, 0, 1)),
                            top_k, thresh=0.01)
        APs = [str(num) for num in APs]
        mAP = str(mAP)
    else:
        test_net(config, test_save_dir, net, config.test['cuda'], testset, num_classes,
                 BaseTransform(config.min_dim, config.MEANS, rgb_std, (2, 0, 1)),
                 top_k, thresh=0.01)


def test_net(config, save_folder, net, cuda, testset, num_classes, transform, max_per_image=300, thresh=0.005):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if config.test['retest']:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return

    for i in range(num_images):
        img = testset.pull_image(i)
        x = Variable(transform(img).unsqueeze(0), volatile=True)
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        softmax = nn.Softmax(dim=-1)
        detect = Detect(num_classes, config.variance, 0, 200, 0.01, 0.45)
        # priorbox = PriorBox(config)
        # priors = Variable(priorbox.forward())
        out = net(x=x)  # forward pass
        boxes, scores = detect(
            out[0],  # loc preds        mbox_loc
            softmax(out[1]),  # conf preds       mbox_conf
            out[2])  # default boxes    mbox_priorbox
        # boxes, scores = out.data
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45, force_cpu=False)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                  .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    if config.test['dataset'] == 'VOC':
        APs, mAP = testset.evaluate_detections(all_boxes, save_folder)
        return APs, mAP
    else:
        testset.evaluate_detections(all_boxes, save_folder)


'''
if __name__ == '__main__':
    test()
'''
