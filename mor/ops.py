import torch
from torch import Tensor
from typing import List

__all__ = ["fake_quantize"]

########################################################################################################################
# MOR Ops
########################################################################################################################

def fake_quantize(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int, clamp_threshold: float, mode: int, sf_type: int) -> List[Tensor]:
    """Quantize the input bf16 Tensor to the output bf16 Tensor."""
    return torch.ops.mor.fake_quantize.default(a, dim_x, dim_y, block_x, block_y, clamp_threshold, mode, sf_type)

def _backward(ctx, grad):
    return grad, None, None, None, None, None, None, None

def _setup_context(ctx, inputs, output):
    pass

torch.library.register_autograd("mor::fake_quantize", _backward, setup_context=_setup_context)


def fake_quantize_no_alloc(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int, clamp_threshold: float, mode: int, sf_type: int, output: Tensor) -> Tensor:
    """Quantize the input bf16 Tensor to the output bf16 Tensor. Use the given output tensor."""
    return torch.ops.mor.fake_quantize_no_alloc.default(a, dim_x, dim_y, block_x, block_y, clamp_threshold, mode, sf_type, output)

def _backward_no_alloc(ctx, grad):
    return grad, None, None, None, None, None, None, None, None

def _setup_context_no_alloc(ctx, inputs, output):
    pass

torch.library.register_autograd("mor::fake_quantize_no_alloc", _backward_no_alloc, setup_context=_setup_context_no_alloc)

########################################################################################################################
# Block Scaling Ops
########################################################################################################################
def fake_quantize_block_scaling(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int, e4m3_threshold: float, ss_mode: int) -> List[Tensor]:
    """Quantize the input bf16 Tensor to the output bf16 Tensor using block scaling with adjusted scaling factor."""
    return torch.ops.mor.fake_quantize_block_scaling.default(a, dim_x, dim_y, block_x, block_y, e4m3_threshold, ss_mode)

def _backward_block_scaling(ctx, grad):
    return grad, None, None, None, None, None, None

def _setup_context_block_scaling(ctx, inputs, output):
    pass

torch.library.register_autograd("mor::fake_quantize_block_scaling", _backward_block_scaling, setup_context=_setup_context_block_scaling)

def fake_quantize_block_scaling_inplace(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int, e4m3_threshold: float, ss_mode: int) -> List[Tensor]:
    """Quantize the input bf16 Tensor to the output bf16 Tensor using block scaling with adjusted scaling factor. Change made to the input Tensor a."""
    return torch.ops.mor.fake_quantize_block_scaling_inplace.default(a, dim_x, dim_y, block_x, block_y, e4m3_threshold, ss_mode)

def _backward_block_scaling_inplace(ctx, grad):
    return grad, None, None, None, None, None, None

def _setup_context_block_scaling_inplace(ctx, inputs, output):
    pass

torch.library.register_autograd("mor::fake_quantize_block_scaling_inplace", _backward_block_scaling_inplace, setup_context=_setup_context_block_scaling_inplace)

########################################################################################################################
# Channel Scaling Ops
########################################################################################################################
def fake_quantize_channel_scaling(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int, e4m3_threshold: float, ss_mode: int) -> List[Tensor]:
    """Quantize the input bf16 Tensor to the output bf16 Tensor using channel scaling with adjusted scaling factor."""
    return torch.ops.mor.fake_quantize_channel_scaling.default(a, dim_x, dim_y, block_x, block_y, e4m3_threshold, ss_mode)

def _backward_channel_scaling(ctx, grad):
    return grad, None, None, None, None, None, None

def _setup_context_channel_scaling(ctx, inputs, output):
    pass

torch.library.register_autograd("mor::fake_quantize_channel_scaling", _backward_channel_scaling, setup_context=_setup_context_channel_scaling)

def fake_quantize_channel_scaling_inplace(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int, e4m3_threshold: float, ss_mode: int) -> List[Tensor]:
    """Quantize the input bf16 Tensor to the output bf16 Tensor using channel scaling with adjusted scaling factor. Change made to the input Tensor a."""
    return torch.ops.mor.fake_quantize_channel_scaling_inplace.default(a, dim_x, dim_y, block_x, block_y, e4m3_threshold, ss_mode)

def _backward_channel_scaling_inplace(ctx, grad):
    return grad, None, None, None, None, None, None

def _setup_context_channel_scaling_inplace(ctx, inputs, output):
    pass

torch.library.register_autograd("mor::fake_quantize_channel_scaling_inplace", _backward_channel_scaling_inplace, setup_context=_setup_context_channel_scaling_inplace)

########################################################################################################################
# E8Mx Ops
########################################################################################################################

def fake_quantize_e8mx(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int, quantize_type: int, scaling_strategy: int) -> Tensor:
    """Quantize the input bf16 Tensor to the output bf16 Tensor."""
    return torch.ops.mor.fake_quantize_e8mx.default(a, dim_x, dim_y, block_x, block_y, quantize_type, scaling_strategy)

def _backward_e8mx(ctx, grad):
    return grad, None, None, None, None, None, None

torch.library.register_autograd("mor::fake_quantize_e8mx", _backward_e8mx, setup_context=_setup_context)


def fake_quantize_e8mx_inplace(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int, quantize_type: int, scaling_strategy: int, output: Tensor) -> None:
    """Quantize the input bf16 Tensor to the output bf16 Tensor. Use the given output tensor."""
    return torch.ops.mor.fake_quantize_e8mx_inplace.default(a, dim_x, dim_y, block_x, block_y, quantize_type, scaling_strategy, output)

########################################################################################################################
# The following are the APIs for testing purpose.
########################################################################################################################
def _default_backward(ctx, grad):
    return grad, None, None, None, None

def _default_setup_context(ctx, inputs, output):
    pass

def quant_dequant_e4m3(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int) -> Tensor:
    """Quantize the input bf16 Tensor to the output bf16 Tensor. Use the given output tensor."""
    return torch.ops.mor.quant_dequant_e4m3.default(a, dim_x, dim_y, block_x, block_y)

torch.library.register_autograd("mor::quant_dequant_e4m3", _default_backward, setup_context=_default_setup_context)


def quant_dequant_e5m2(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int) -> Tensor:
    """Quantize the input bf16 Tensor to the output bf16 Tensor. Use the given output tensor."""
    return torch.ops.mor.quant_dequant_e5m2.default(a, dim_x, dim_y, block_x, block_y)

torch.library.register_autograd("mor::quant_dequant_e5m2", _default_backward, setup_context=_default_setup_context)


def quant_dequant_e2m3(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int) -> Tensor:
    """Quantize the input bf16 Tensor to the output bf16 Tensor. Use the given output tensor."""
    return torch.ops.mor.quant_dequant_e2m3.default(a, dim_x, dim_y, block_x, block_y)

torch.library.register_autograd("mor::quant_dequant_e2m3", _default_backward, setup_context=_default_setup_context)


def quant_dequant_e3m2(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int) -> Tensor:
    """Quantize the input bf16 Tensor to the output bf16 Tensor. Use the given output tensor."""
    return torch.ops.mor.quant_dequant_e3m2.default(a, dim_x, dim_y, block_x, block_y)

torch.library.register_autograd("mor::quant_dequant_e3m2", _default_backward, setup_context=_default_setup_context)


def e4m3_with_e8_scale_rne(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int) -> Tensor:
    """Quantize the input bf16 Tensor to the output bf16 Tensor. Use the given output tensor."""
    return torch.ops.mor.e4m3_with_e8_scale_rne.default(a, dim_x, dim_y, block_x, block_y)

torch.library.register_autograd("mor::e4m3_with_e8_scale_rne", _default_backward, setup_context=_default_setup_context)


def e4m3_with_e8_scale_rz(a: Tensor, dim_x: int, dim_y: int, block_x: int, block_y: int) -> Tensor:
    """Quantize the input bf16 Tensor to the output bf16 Tensor. Use the given output tensor."""
    return torch.ops.mor.e4m3_with_e8_scale_rz.default(a, dim_x, dim_y, block_x, block_y)

torch.library.register_autograd("mor::e4m3_with_e8_scale_rz", _default_backward, setup_context=_default_setup_context)
