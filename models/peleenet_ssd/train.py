from .data import *
from .utils.augmentations import SSDAugmentation
from .layers.modules import MultiBoxLoss
from .peleenet_ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np


def train(config, model):
    if config.name == 'COCO':
        from .data.coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, get_label_map
    elif config.name == 'VOC':
        from .data.voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES

    if not os.path.exists(config.train['save_folder']):
        os.mkdir(config.train['save_folder'])

    if config.train['dataset'] == 'COCO':
        dataset = COCODetection(root=config.train['dataset_root'],
                                transform=SSDAugmentation(config.min_dim, config.MEANS))
    elif config.train['dataset'] == 'VOC':
        dataset = VOCDetection(root=config.train['dataset_root'],
                               transform=SSDAugmentation(config.min_dim, config.MEANS))

    if config.train['visdom']:
        import visdom
        viz = visdom.Visdom()

    if config.train['cuda']:
        net = torch.nn.DataParallel(model.cuda())
        cudnn.benchmark = True

    if config.train['resume']:
        print('Resuming training, loading {}...'.format(config.train['resume']))
        # model.load_weights(config.train['resume'])

    optimizer = optim.SGD(net.parameters(), lr=config.train['lr'], momentum=config.train['momentum'],
                          weight_decay=config.train['weight_decay'])
    criterion = MultiBoxLoss(config.variance, config.num_classes, 0.5, True, 0, True, 3, 0.5,
                             False, config.train['cuda'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config.train['milestones'])
    epochs = config.train['milestones'][-1]

    net.train()
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // config.train['batch_size']
    print('Training SSD on:', dataset.name)

    step_index = 0

    if config.train['visdom']:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, config.train['batch_size'],
                                  num_workers=config.train['num_workers'],
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    print(len(dataset))
    global_step = 0
    for epoch in range(epochs):
        print(config.train['save_folder'] + '' + config.train['dataset'] + str(epoch) + '.pth')
        for step, (images, targets) in enumerate(data_loader):
            if config.train['cuda']:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                images = images
                targets = [ann for ann in targets]
            # forward
            t0 = time.time()
            out = net(images)
            # backprop
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            if step > 0 and step % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('epoch' + repr(epoch) + ' iter ' + repr(step) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

            # if config.train['visdom']:
            #     update_vis_plot(iteration, loss_l.data[0], loss_c.data[0], iter_plot, epoch_plot, 'append')

            if step != 0 and step % 5000 == 0:
                print('Saving state, iter:', global_step)
                torch.save(model.state_dict(), 'weights/peleenet_ssd800_COCO_' +
                           repr(global_step) + '.pth')
            global_step += 1
        # adjust lr
        lr_scheduler.step()
        torch.save(model.state_dict(), config.train['save_folder'] + '' + config.train['dataset'] + str(epoch) + '.pth')


'''
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(config.train['start_iter'], config.max_iter):
        if config.train['visdom'] and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in config.lr_steps:
            step_index += 1
            adjust_learning_rate(optimizer, config.train['gamma'], step_index)

        # load train data
        images, targets = next(batch_iterator)

        if config.train['cuda']:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = images
            targets = [ann for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

        if config.train['visdom']:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0], iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(model.state_dict(), 'weights/peleenet_ssd800_COCO_' +
                       repr(iteration) + '.pth')
    torch.save(model.state_dict(),
               config.train['save_folder'] + '' + config.train['dataset'] + '.pth')
'''


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = config.train['lr'] * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


'''
if __name__ == '__main__':
    train()
'''
