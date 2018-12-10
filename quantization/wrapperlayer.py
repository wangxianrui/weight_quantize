import torch
from . import quant_util

"""
    quantize weights and bias
"""


class WrapperLayer(torch.nn.Module):
    """
        only support nn.Conv2d, nn.Linear now
        bits_params 8bits, 4bits, 2bits  quantize both weights and biaa
        w           * input + b                                 = out
        w_scale               b_scale                             out_scale
        w * w_scale * input + b * b_scale * (w_scale / b_scale) = out * out_scael * (w_scale / out_scael)
        w * w_scale * input + b * w_scale                       = out * w_scale
    """

    def __init__(self, module, bits_params):
        super(WrapperLayer, self).__init__()
        self.quantized_module = module
        self.bits_params = bits_params
        self.quant_range = quant_util.get_quantized_range(bits_params, signed=True)

        # quantize weight
        weights_max = quant_util.get_tensor_max_abs(module.weight)
        w_scale = quant_util.symmetric_linear_quantization_scale_factor(bits_params, weights_max)
        module.weight.data = quant_util.linear_quantize_clamp(module.weight, w_scale, self.quant_range[0],
                                                              self.quant_range[1])
        # quantize bias
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data = quant_util.linear_quantize_clamp(module.bias, w_scale, self.quant_range[0],
                                                                self.quant_range[1])
        self.register_buffer('w_scale', torch.tensor(w_scale).to(device=module.weight.data.device))

    def forward(self, input):
        # prepare
        if self.training:
            self._prepare()
        # forward
        out = self.quantized_module.forward(input) / self.w_scale
        return out

    def _prepare(self):
        # quantize weight
        weights_max = quant_util.get_tensor_max_abs(self.quantized_module.weight)
        w_scale = quant_util.symmetric_linear_quantization_scale_factor(self.bits_params, weights_max)
        self.quantized_module.weight.data = quant_util.linear_quantize_clamp(self.quantized_module.weight, w_scale,
                                                                             self.quant_range[0],
                                                                             self.quant_range[1])
        # quantize bias
        if hasattr(self.quantized_module, 'bias') and self.quantized_module.bias is not None:
            self.quantized_module.bias.data = quant_util.linear_quantize_clamp(self.quantized_module.bias, w_scale,
                                                                               self.quant_range[0],
                                                                               self.quant_range[1])
        self.w_scale *= w_scale
