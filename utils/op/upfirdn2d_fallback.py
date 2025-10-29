# utils/op/upfirdn2d_fallback.py
import torch
import torch.nn.functional as F

def _native(
        x: torch.Tensor,
        kernel: torch.Tensor,
        up_x: int = 1, up_y: int = 1,
        down_x: int = 1, down_y: int = 1,
        pad_x0: int = 0, pad_x1: int = 0,
        pad_y0: int = 0, pad_y1: int = 0) -> torch.Tensor:
    """Pure‑PyTorch UpFirDn (GPU/CPU, no custom ops)."""
    b, c, in_h, in_w = x.shape
    x = x.reshape(-1, in_h, in_w, 1)

    # 1. Upsample by inserting zeros
    x = x.view(-1, in_h, 1, in_w, 1, 1)
    x = F.pad(x, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    x = x.view(-1, in_h * up_y, in_w * up_x, 1)

    # 2. Pad / crop
    x = F.pad(
        x,
        [0, 0,
         max(pad_x0, 0), max(pad_x1, 0),
         max(pad_y0, 0), max(pad_y1, 0)],
    )
    x = x[
        :,
        max(-pad_y0, 0) : x.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : x.shape[2] - max(-pad_x1, 0),
        :,
    ]

    # 3. Convolution with flipped kernel
    x = x.permute(0, 3, 1, 2)           # (N,1,H,W)
    k = torch.flip(kernel, [0, 1]).view(1, 1, *kernel.shape).to(x.device)
    x = F.conv2d(x, k)

    # 4. Downsample
    x = x[..., ::down_y, ::down_x]       # keep stride
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel.shape[0]) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel.shape[1]) // down_x + 1
    x = x.view(b, c, out_h, out_w)
    return x

def upfirdn2d(
        x: torch.Tensor,
        kernel: torch.Tensor,
        up: int | tuple[int, int] = 1,
        down: int | tuple[int, int] = 1,
        pad: tuple[int, int] | tuple[int, int, int, int] = (0, 0)
) -> torch.Tensor:
    """Wrapper that mimics the signature of the original CUDA op."""
    up_x, up_y = (up, up) if isinstance(up, int) else up
    down_x, down_y = (down, down) if isinstance(down, int) else down
    if len(pad) == 2:
        pad_x0, pad_x1 = pad
        pad_y0, pad_y1 = pad
    else:
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
    return _native(x, kernel, up_x, up_y, down_x, down_y,
                   pad_x0, pad_x1, pad_y0, pad_y1)
