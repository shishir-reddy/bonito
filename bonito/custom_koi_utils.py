import torch
from koi.utils import ffi
from collections import namedtuple


Buffer = namedtuple('Buffer', 'data ptr')


def void_ptr(x):
    """
    Return a void * for given Tensor `x`.
    """

    # Try sending tensor to CPU first
    return ffi.cast("void *", x.to(torch.device('cpu')).data_ptr())
    # return ffi.cast("void *", x.data_ptr())


def empty(size, device, dtype=torch.float16):
    """
    Create an empty Tensor of size `size` on device `device`.
    """
    x = torch.empty(size, dtype=dtype, device=device)
    return Buffer(x, void_ptr(x))


def zeros(size, device, dtype=torch.float16):
    """
    Create an zeros Tensor of size `size` on device `device`.
    """
    x = torch.zeros(size, dtype=dtype, device=device)
    return Buffer(x, void_ptr(x))


def quantize_tensor(tensor, levels=256, dim=0, z=None):
    """
    Quantize a tensor to int8, returning the per-channel scales and the quantized tensor.

    If z is provided, the floating point range used for quantisation is clipped to
    z times the standard deviation from the mean for each channel.
    """

    fp_max = torch.abs(tensor.max(dim=dim)[0])
    fp_min = torch.abs(tensor.min(dim=dim)[0])
    fp_range = torch.cat((fp_min[:,None], fp_max[:,None]), 1).max(dim=1)[0] * 2

    if z is not None:
        fp_mean = tensor.mean(axis=0)
        fp_std = tensor.std(axis=0)
        fp_range_z = 2 * (abs(fp_mean) + fp_std * z)
        fp_range = torch.min(fp_range, fp_range_z)

    quantization_scale = levels / fp_range
    quantization_max = (levels / 2) - 1
    tensor_quantized  = (tensor * quantization_scale).round().clip(-quantization_max, quantization_max)
    tensor_quantized = tensor_quantized.type(torch.int8)

    return  quantization_scale.float(), tensor_quantized
