import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .layers import *
from .peleenet import *


# import config as cfg

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base PeleeNet network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: PeleeNet layers for input, size of either 750 or 800
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, config, base, extras, head):
        super(SSD, self).__init__()
        #self.phase = config.phase
        self.num_classes = config.num_classes
        self.priorbox = PriorBox(config)
        self.priors = Variable(self.priorbox.forward())

        self.size = config.min_dim

        # SSD network
        self.peleenet = base
        # print("peleenet:")
        # print(self.peleenet)
        # Layer learns to scale the l2 normalized features from conv4_3
        # self.L2Norm = L2Norm(512, 20)
        # self.extras = extras

        self.stage4_tb_ext_pm2 = extras['stage4_tb/ext/pm2']
        self.stage4_tb_ext_pm3 = extras['stage4_tb/ext/pm3']
        self.stage4_tb_ext_pm4 = extras['stage4_tb/ext/pm4']
        self.stage4_tb_ext_pm5 = extras['stage4_tb/ext/pm5']
        self.stage4_tb_ext_pm6 = extras['stage4_tb/ext/pm6']

        self.stage4_tb_ext_pm2_ = nn.Sequential(extras['stage4_tb/ext/pm2/b2a'],
                                                extras['stage4_tb/ext/pm2/b2b'],
                                                extras['stage4_tb/ext/pm2/b2c'])
        self.stage4_tb_ext_pm3_ = nn.Sequential(extras['stage4_tb/ext/pm3/b2a'],
                                                extras['stage4_tb/ext/pm3/b2b'],
                                                extras['stage4_tb/ext/pm3/b2c'])
        self.stage4_tb_ext_pm4_ = nn.Sequential(extras['stage4_tb/ext/pm4/b2a'],
                                                extras['stage4_tb/ext/pm4/b2b'],
                                                extras['stage4_tb/ext/pm4/b2c'])
        self.stage4_tb_ext_pm5_ = nn.Sequential(extras['stage4_tb/ext/pm5/b2a'],
                                                extras['stage4_tb/ext/pm5/b2b'],
                                                extras['stage4_tb/ext/pm5/b2c'])
        self.stage4_tb_ext_pm6_ = nn.Sequential(extras['stage4_tb/ext/pm6/b2a'],
                                                extras['stage4_tb/ext/pm6/b2b'],
                                                extras['stage4_tb/ext/pm6/b2c'])
        # print("stage4_tb_ext_pm6_:")
        # print(self.stage4_tb_ext_pm6_)

        self.stage4_tb_relu_ext1_fe1_1_fe1_2 = nn.Sequential(nn.Conv2d(704, 256, 1, 1, 0),
                                                             nn.ReLU(inplace=True),
                                                             nn.Conv2d(256, 256, 3, 2, 1),
                                                             nn.ReLU(inplace=True))
        self.ext1_fe2_1_fe2_2 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(128, 256, 3, 1, 0),
                                              nn.ReLU(inplace=True))
        self.ext1_fe3_1_fe3_2 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(128, 256, 3, 1, 0),
                                              nn.ReLU(inplace=True))

        # detection head        
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        # print("loc:")
        # print(self.loc)
        # print("conf:")
        # print(self.conf)

        #if self.phase == 'test':
        #self.softmax = nn.Softmax(dim=-1)
        #self.detect = Detect(self.num_classes, config.variance, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test_bc:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        for name, module in next(self.peleenet.children()).named_children():
            if name == 'stage_3':
                # print(name)
                for ids, module1 in enumerate(module):
                    # print(ids)
                    if ids == 8:
                        x = module1(x)
                        stage3_tb_out = x
                        stage4_tb_ext_pm2_out = self.stage4_tb_ext_pm2(stage3_tb_out)
                        stage4_tb_ext_pm2__out = self.stage4_tb_ext_pm2_(stage3_tb_out)
                        stage4_tb_ext_pm2_res_out = stage4_tb_ext_pm2_out + stage4_tb_ext_pm2__out
                        sources.append(stage4_tb_ext_pm2_res_out)
                        sources.append(stage4_tb_ext_pm2_res_out)

                    else:
                        x = module1(x)

            else:
                x = module(x)

        stage4_tb_ext_pm3_out = self.stage4_tb_ext_pm3(x)
        stage4_tb_ext_pm3__out = self.stage4_tb_ext_pm3_(x)
        stage4_tb_ext_pm3_res_out = stage4_tb_ext_pm3_out + stage4_tb_ext_pm3__out
        sources.append(stage4_tb_ext_pm3_res_out)

        stage4_tb_relu_ext1_fe1_1_fe1_2_out = self.stage4_tb_relu_ext1_fe1_1_fe1_2(x)
        stage4_tb_ext_pm4_out = self.stage4_tb_ext_pm4(stage4_tb_relu_ext1_fe1_1_fe1_2_out)
        stage4_tb_ext_pm4__out = self.stage4_tb_ext_pm4_(stage4_tb_relu_ext1_fe1_1_fe1_2_out)
        stage4_tb_ext_pm4_res_out = stage4_tb_ext_pm4_out + stage4_tb_ext_pm4__out
        sources.append(stage4_tb_ext_pm4_res_out)

        ext1_fe2_1_fe2_2_out = self.ext1_fe2_1_fe2_2(stage4_tb_relu_ext1_fe1_1_fe1_2_out)
        stage4_tb_ext_pm5_out = self.stage4_tb_ext_pm5(ext1_fe2_1_fe2_2_out)
        stage4_tb_ext_pm5__out = self.stage4_tb_ext_pm5_(ext1_fe2_1_fe2_2_out)
        stage4_tb_ext_pm5_res_out = stage4_tb_ext_pm5_out + stage4_tb_ext_pm5__out
        sources.append(stage4_tb_ext_pm5_res_out)

        ext1_fe3_1_fe3_2_out = self.ext1_fe3_1_fe3_2(ext1_fe2_1_fe2_2_out)
        stage4_tb_ext_pm6_out = self.stage4_tb_ext_pm6(ext1_fe3_1_fe3_2_out)
        stage4_tb_ext_pm6__out = self.stage4_tb_ext_pm6_(ext1_fe3_1_fe3_2_out)
        stage4_tb_ext_pm6_res_out = stage4_tb_ext_pm6_out + stage4_tb_ext_pm6__out
        sources.append(stage4_tb_ext_pm6_res_out)

        '''                        
        # apply vgg up to conv4_3 relu
        #for k in range(23):
            #x = self.vgg[k](x)
        x = self.peleenet
        #sources.append(x)

        #s = self.L2Norm(x)
        #sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        '''

        # apply multibox head to source layers  here is the Permute layer after loc and conf
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        '''
        print(len(sources))
        print("loc")
        for tmp in loc:
            print(tmp.view(tmp.size(0),-1).size())

        print("conf")
        for tmp in conf:
            print(tmp.view(tmp.size(0),-1).size())
        '''

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # print("self.priorbox:")
        # print(self.priors.size())
        # print(conf.size())
        # print(loc.size())
        '''
        if self.phase == "test":
            boxes,scores = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds        mbox_loc
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds       mbox_conf
                self.priors.type(type(x.data))  # default boxes    mbox_priorbox
            )
            return boxes,scores
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors.to(device=loc.device)
            )
            # print(self.priors)
            return output
        '''
        output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors.to(device=loc.device)
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


'''
# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers
'''


def add_extras(config):
    # Extra layers added to PeleeNet for feature scaling
    layers = {}
    for i, v in enumerate(config):
        name1 = "stage4_tb/ext/pm" + str(i + 2)
        layers[name1] = Conv_bn_relu(v, 256, 1, 1, 0, use_relu=False)
        name2_1 = name1 + "/b2a"
        layers[name2_1] = Conv_bn_relu(v, 128, 1, 1, 0)
        name2_2 = name1 + "/b2b"
        layers[name2_2] = Conv_bn_relu(128, 128, 3, 1, 1)
        name2_3 = name1 + "/b2c"
        layers[name2_3] = Conv_bn_relu(128, 256, 1, 1, 0, use_relu=False)
        # name_sum = name1 + "/res"
        # layers[name_sum] = layers[name1](Eltwise(layers[name2_3]))
    # print(layers['stage4_tb/ext/pm3/b2a'])
    return layers


def multibox(config, num_classes):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(config):
        # name_loc = "ext/pm" + str(k+1) + "_mbox_loc"
        loc_layers += [nn.Conv2d(256, v * 4, kernel_size=1, padding=0)]
        # name_conf = "ext/pm" + str(k+1) + "_mbox_conf"
        conf_layers += [nn.Conv2d(256, v * num_classes, kernel_size=1, padding=0)]
        # print(name_loc)
        # print(name_conf)
    '''
    for name,module in peleenet.named_children():
        print(name)
        #print(module.named_children())
        for sub_name,sub_module in module.named_children():
            print(sub_name)
            if sub_name in ['stage_3','stage_4']:
                print(sub_module)
        #print(module)
        
    
    priorbox_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):                                    # (0,21) (1,-2)
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
        #priorbox_layers += 
    for k, v in enumerate(extra_layers[1::2], 2):                         # (0,extra_layer(1)) (1,extra_layer(3) (2,extra_layer(5) (3,extra_layer(7) (4,2)
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    '''

    return (loc_layers, conf_layers)


'''
base = {
    '800': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
'''

extras = {
    '800': [512, 704, 256, 256, 256],
    '750': [512, 704, 256, 256, 256],
}
mbox = {
    '800': [6, 6, 6, 6, 6, 6],  # number of boxes per feature map location
    '750': [6, 6, 6, 6, 6, 6],
}


def build_ssd(config):
    #phase = config.phase
    size = config.min_dim
    num_classes = config.num_classes
    #if phase != "test" and phase != "train":
        #print("ERROR: Phase: " + phase + " not recognized")
        #return
    if size != 800:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD800 (size=800) is supported!")
        return
    # base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
    # add_extras(extras[str(size)], 1024),
    # mbox[str(size)], num_classes)
    peleenet = PeleeNet(num_classes)
    # print("PeleeNet:")
    # print(peleenet)
    extra_ = add_extras(extras[str(size)])
    head_ = multibox(mbox[str(size)], num_classes)

    model = SSD(config, peleenet, extra_, head_)
    # x = torch.rand(1,3,800,800)
    # model(x)
    return model


if __name__ == '__main__':
    import config as cfg

    ssd_net = build_ssd(cfg)
    input = torch.rand(4, 3, 800, 800)
    output = ssd_net(input)

    torch.save(ssd_net.state_dict(), 'peleenet.pth')
    ssd_net.load_state_dict(torch.load('/home/wxrui/Pytorch_PeleeNet_SSD/zzzzzzz/transedmodel.pth'))
