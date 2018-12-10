import torch
import torch.utils.data
import models.yolo.data.coco_dataset as coco_dataset
from ..yolo import config
from ..yolo import yolo_loss
from tensorboardX import SummaryWriter
import quantization.quant_util as quant_util


def yolo_train(model):
    # DataLoader
    dataloader = torch.utils.data.DataLoader(
        coco_dataset.COCODataset(config.data_root, config.ModelTrain.train_path, (config.img_w, config.img_h),
                                 is_training=True), batch_size=config.ModelTrain.batch_size, shuffle=True,
        pin_memory=True)

    # optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.ModelTrain.lr, momentum=config.ModelTrain.momentum,
                                weight_decay=config.ModelTrain.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config.ModelTrain.milestones)

    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            yolo_loss.YOLOLoss(config.anchors[i], config.num_classes, (config.img_w, config.img_h)))

    print('Start training...')

    model.train()
    global_step = 0
    writer = SummaryWriter('log')
    for epoch in range(config.ModelTrain.epochs):
        for step, samples in enumerate(dataloader):
            images, labels = samples["image"], samples["label"]
            outputs = model(images)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = [[] for i in range(len(losses_name))]
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]

            if step > 0 and step % 10 == 0:
                _loss = loss.item()
                lr = optimizer.param_groups[0]['lr']
                print("epoch [%.3d] iter = %d loss = %.2f  lr = %.5f " % (epoch, step, _loss, lr))
                for i, name in enumerate(losses_name):
                    value = _loss if i == 0 else losses[i]
                    writer.add_scalar(name, value, global_step)
            if step > 0 and step % 1000 == 0:
                print('save quantized model parameters', global_step)
                quant_util.save_quantized_model(model, 'output/quantized_%d.pth' % global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        lr_scheduler.step()
