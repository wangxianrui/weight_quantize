import os
import json
import torch
import models.yolo.data.coco_dataset as coco_dataset
from ..yolo import config
from ..yolo import yolo_loss
from ..yolo import utils


def yolo_eval(model):
    dataloader = torch.utils.data.DataLoader(
        coco_dataset.COCODataset(config.data_root, config.ModelEval.eval_path, (config.img_w, config.img_h),
                                 is_training=False), batch_size=config.ModelEval.batch_size, shuffle=False)
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            yolo_loss.YOLOLoss(config.anchors[i], config.num_classes, (config.img_w, config.img_h)))
    # Start inference
    json_result = list()
    labelmap = utils.get_labelmap()
    model.eval()
    for step, samples in enumerate(dataloader):
        print('[%d/%d]' % (step, len(dataloader)))
        images, targets = samples["image"].cuda(), samples["label"].cuda()
        images_id = samples["img_id"].numpy()
        height = samples['height'].numpy()
        width = samples['width'].numpy()

        # inference
        outputs = model(images)
        output_list = []
        for i in range(3):
            output_list.append(yolo_losses[i](outputs[i]))
        # [batchsize, T, 85]
        output = torch.cat(output_list, 1)
        # [batchsize, T, 7]  x1, y1, x2, y2, conf, cls_score, cls
        detections = utils.non_max_suppression(output, config.num_classes, conf_thres=config.ModelEval.conf_thres,
                                               nms_thres=config.ModelEval.nms_thres)
        # format result
        batch_detections = list()
        for detection in detections:
            if detection is None:
                batch_detections.append(None)
                continue
            # top K
            k = min(detection.size()[0], config.ModelEval.bbox_per)
            _, index = torch.topk(detection, k, 0)
            detection = detection[index[:, 4]]
            # x1,y1,x2,y2 convert to x1,y1,width,height
            detection[:, 2] -= detection[:, 0]
            detection[:, 3] -= detection[:, 1]
            # normalize
            detection = detection.cpu() \
                        / torch.Tensor([config.img_w, config.img_h, config.img_w, config.img_h, 1, 1, 1])
            batch_detections.append(detection.numpy())

        # write result
        assert len(batch_detections) == len(width) == len(height) == len(images_id)
        for bt_id in range(len(batch_detections)):
            if batch_detections[bt_id] is None:
                continue
            for bbox_id in range(len(batch_detections[bt_id])):
                value = int(batch_detections[bt_id][bbox_id][-1]) + 1
                key = int(list(labelmap.keys())[list(labelmap.values()).index(value)])
                pred = {
                    "image_id": int(images_id[bt_id]),
                    "category_id": key,
                    "bbox": [
                        round(batch_detections[bt_id][bbox_id][0] * width[bt_id], 2),
                        round(batch_detections[bt_id][bbox_id][1] * height[bt_id], 2),
                        round(batch_detections[bt_id][bbox_id][2] * width[bt_id], 2),
                        round(batch_detections[bt_id][bbox_id][3] * height[bt_id], 2)
                    ],
                    "score": float(batch_detections[bt_id][bbox_id][4]) * float(batch_detections[bt_id][bbox_id][5])
                }
                json_result.append(pred)
    json.dump(json_result, open('result.json', 'w'), indent=4)
    os.system('python cocoevaldemo.py ')
