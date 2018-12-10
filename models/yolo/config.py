anchors = [[[116, 90], [156, 198], [373, 326]],
           [[30, 61], [62, 45], [59, 119]],
           [[10, 13], [16, 30], [33, 23]]]
num_classes = 80
img_w = 416
img_h = 416
data_root = '/home/wxrui/DATA/coco'


class ModelTrain:
    lr = 1e-4
    milestones = [5, 10, 15, 20]
    epochs = milestones[-1]
    weight_decay = 1e-4
    momentum = 0.9
    batch_size = 8
    train_path = "train.txt"


class ModelEval:
    batch_size = 1
    conf_thres = 0.3
    nms_thres = 0.5
    bbox_per = 100
    eval_path = "minival2014.txt"
