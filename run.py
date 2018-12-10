import torch
import quantization.quant_util as quant_util
import quantization.quantizer as quantizer


def yolo():
    from models.yolo import dark_yolo
    from models.yolo import yolo_train
    from models.yolo import yolo_eval

    # create model
    model = dark_yolo.Dark_YOLO()
    model = torch.nn.DataParallel(model.cuda())

    # load checkpoint
    checkpoint = 'models/yolo/yolo.pth'
    model.load_state_dict(torch.load(checkpoint))

    # quantize model
    quantizer.quantize_model(model, 8)

    # load  quantized checkpoint
    # quant_util.load_quantized_model(model, 'output/quantized_294402.pth')

    # train model
    # yolo_train.yolo_train(model)

    # eval model
    yolo_eval.yolo_eval(model)


def ssd():
    from models.ssd.ssd import build_ssd
    from models.ssd.data import VOCAnnotationTransform, VOCDetection, BaseTransform
    from models.ssd.eval import test_net

    # create model
    num_classes = 21
    model = build_ssd('test', 300, num_classes)
    model = torch.nn.DataParallel(model.cuda())

    # load checkpoint
    checkpoint = 'models/ssd/weights/ssd300_mAP_77.43_v2.pth'
    model.module.load_state_dict(torch.load(checkpoint))

    # quantize model
    quantizer.quantize_model(model, 8)

    # load quantized checkpoint
    # quant_util.load_quantized_model(model, 'quantized_ssd.pth')

    # eval model
    voc_root = '/home/wxrui/DATA/VOCdevkit/'
    dataset_mean = (104, 117, 123)
    dataset = VOCDetection(voc_root, [('2007', 'test')],
                           BaseTransform(300, dataset_mean),
                           VOCAnnotationTransform())
    test_net(model, dataset)


def mobile_ssd():
    import models.mobile_ssd.lib.ssds_train as ssds_train
    from models.mobile_ssd.lib.utils.config_parse import cfg_from_file

    cfg = cfg_from_file('models/mobile_ssd/experiments/cfgs/ssd_lite_mobilenetv2_train_coco.yml')
    s = ssds_train.Solver(cfg)

    quantizer.quantize_model(s.model, 8)

    s.test_model()


def peleenet_ssd():
    from models.peleenet_ssd import config
    from models.peleenet_ssd.peleenet_ssd import build_ssd
    from models.peleenet_ssd.train import train
    from models.peleenet_ssd.test import test
    # create model
    model = build_ssd(config)
    model.load_state_dict(torch.load(config.test['trained_model']))

    quantizer.quantize_model(model, 8)

    # train
    train(config, model)

    # test
    test(config, model)


if __name__ == '__main__':
    peleenet_ssd()
