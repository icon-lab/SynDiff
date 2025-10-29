#from .fused_act import FusedLeakyReLU, fused_leaky_relu
#import torch
#import torch.nn.functional as F
#from .upfire2d_fallback import upfirdn2d

try:
    # Keep the original path in case you later add CUDA/ninja
    from .upfirdn2d import upfirdn2d   # ← will attempt to compile
except Exception:
    # No CUDA_HOME / ninja / nvcc?  Fall back!
    from .upfirdn2d_fallback import upfirdn2d


try:
    from .fused_act import FusedLeakyReLU, fused_leaky_relu
except Exception:
    import torch
    import torch.nn.functional as F
    class FusedLeakyReLU(torch.nn.Module):
        def __init__(self, channels, bias=True, negative_slope=0.2):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(channels)) if bias else None
            self.neg_slope = negative_slope

        def forward(self, x):
            if self.bias is not None:
                x = x + self.bias.view(1, -1, 1, 1)
            return F.leaky_relu(x, self.neg_slope)

    def fused_leaky_relu(x, bias, negative_slope=0.2):
        return F.leaky_relu(x + bias.view(1, -1, 1, 1), negative_slope)


#from .fused_act import FusedLeakyReLU, fused_leaky_relu
#from .upfirdn2d import upfirdn2d
