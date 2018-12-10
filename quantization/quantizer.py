import torch
from . import wrapperlayer


def quantize_model(model, bits_params):
    quantize_list = [torch.nn.Conv2d, torch.nn.Linear]
    for name, module in model.named_children():
        if type(module) in quantize_list:
            model.add_module(name, replace_module(module, bits_params))
        if has_children(module):
            quantize_model(module, bits_params)


def replace_module(module, bits_params):
    return wrapperlayer.WrapperLayer(module, bits_params)


def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False
