import torch


def symmetric_linear_quantization_scale_factor(num_bits, saturation_val):
    # Leave one bit for sign
    n = 2 ** (num_bits - 1) - 1
    return n / saturation_val


def linear_quantize(input, scale_factor):
    return torch.round(scale_factor * input)


def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max):
    output = linear_quantize(input, scale_factor)
    return torch.clamp(output, clamp_min, clamp_max)


def get_tensor_max_abs(tensor):
    return max(abs(tensor.max().item()), abs(tensor.min().item()))


def get_quantized_range(num_bits, signed=True):
    if signed:
        n = 2 ** (num_bits - 1)
        return [-n, n - 1]
    return [0, 2 ** num_bits - 1]


def save_quantized_model(model, filename):
    checkpoint = model.state_dict()
    for name in checkpoint.keys():
        if 'quantized_module' in name:
            checkpoint[name] = checkpoint[name].to(torch.int8)
    torch.save(checkpoint, filename)


def load_quantized_model(model, filename):
    checkpoint = torch.load(filename)
    for name in checkpoint.keys():
        if 'quantized_module' in name:
            checkpoint[name] = checkpoint[name].to(torch.float32)
    model.load_state_dict(checkpoint)
