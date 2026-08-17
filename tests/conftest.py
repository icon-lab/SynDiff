import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def small_config():
    """A tiny NCSNpp/diffusion config so the tests stay fast on CPU."""
    return argparse.Namespace(
        image_size=32, num_channels=2, centered=True, num_channels_dae=8,
        n_mlp=2, ch_mult=[1, 2], num_res_blocks=1, attn_resolutions=(16,),
        dropout=0., resamp_with_conv=True, conditional=True, fir=True,
        fir_kernel=[1, 3, 3, 1], skip_rescale=True, resblock_type='biggan',
        progressive='none', progressive_input='residual',
        progressive_combine='sum', embedding_type='positional',
        fourier_scale=16., not_use_tanh=False, nz=8, z_emb_dim=16,
        t_emb_dim=16, ngf=8, num_timesteps=4, beta_min=0.1, beta_max=20.,
        use_geometric=False,
    )
