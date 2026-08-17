"""The CUDA kernels and their pure-PyTorch fallbacks must agree.

utils/op builds its extensions lazily and falls back when the toolchain is
missing, so these tests only run where the kernels are actually available.
"""
import sys

import pytest
import torch

import utils.op  # noqa: F401  (populates sys.modules with the submodules)

upfirdn2d_module = sys.modules['utils.op.upfirdn2d']
fused_act_module = sys.modules['utils.op.fused_act']

needs_kernels = pytest.mark.skipif(
    not torch.cuda.is_available()
    or upfirdn2d_module.upfirdn2d_op is None
    or fused_act_module.fused is None,
    reason="CUDA extensions are not built on this machine",
)


def fir_kernel(device):
    k = torch.tensor([1., 3., 3., 1.])
    k = torch.outer(k, k)
    return (k / k.sum()).to(device)


def test_fallback_is_selected_when_the_extension_is_missing(monkeypatch):
    """A CPU tensor must never reach the CUDA path."""
    monkeypatch.setattr(upfirdn2d_module, 'UpFirDn2d', None)
    out = upfirdn2d_module.upfirdn2d(torch.randn(1, 1, 8, 8),
                                     fir_kernel('cpu'), up=1, down=1, pad=(1, 1))
    assert out.shape == (1, 1, 7, 7)


@needs_kernels
@pytest.mark.parametrize("up,down,pad", [(2, 1, (2, 1)), (1, 2, (1, 1)), (1, 1, (1, 1))])
def test_upfirdn2d_matches_the_native_implementation(up, down, pad):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 16, device='cuda')
    k = fir_kernel('cuda')
    cuda_out = upfirdn2d_module.UpFirDn2d.apply(
        x, k, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1]))
    native_out = upfirdn2d_module.upfirdn2d_native(
        x, k, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    assert cuda_out.shape == native_out.shape
    assert torch.allclose(cuda_out, native_out, atol=1e-6)


@needs_kernels
def test_upfirdn2d_gradients_match_the_native_implementation():
    torch.manual_seed(0)
    k = fir_kernel('cuda')
    a = torch.randn(2, 3, 16, 16, device='cuda', requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    upfirdn2d_module.UpFirDn2d.apply(a, k, (2, 2), (1, 1), (2, 1, 2, 1)).square().sum().backward()
    upfirdn2d_module.upfirdn2d_native(b, k, 2, 2, 1, 1, 2, 1, 2, 1).square().sum().backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-5)


@needs_kernels
def test_fused_leaky_relu_matches_the_native_implementation():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 16, 16, device='cuda')
    bias = torch.randn(8, device='cuda')
    scale = 2 ** 0.5
    cuda_out = fused_act_module.FusedLeakyReLUFunction.apply(x, bias, 0.2, scale)
    native_out = torch.nn.functional.leaky_relu(
        x + bias.view(1, -1, 1, 1), negative_slope=0.2) * scale
    assert torch.allclose(cuda_out, native_out, atol=1e-6)
