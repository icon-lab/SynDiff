"""Shape and gradient smoke tests for the networks, small enough to run on CPU."""
import pytest
import torch
import torch.nn as nn

import backbones.generator_resnet as generator_resnet
from backbones.discriminator import Discriminator_large, Discriminator_small
from backbones.ncsnpp_generator_adagn import NCSNpp


def test_ncsnpp_roundtrip_shape_and_backward(small_config):
    net = NCSNpp(small_config)
    x = torch.randn(2, small_config.num_channels,
                    small_config.image_size, small_config.image_size)
    t = torch.randint(0, small_config.num_timesteps, (2,))
    z = torch.randn(2, small_config.nz)

    out = net(x, t, z)
    assert out.shape == x.shape
    # the default config keeps the tanh, so outputs stay in [-1, 1]
    assert out.abs().max() <= 1.0

    out.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in net.parameters())


def test_untrained_output_is_near_zero(small_config):
    """NCSN++ zero-initialises its residual and output convolutions on purpose.

    Pinning this down documents why an untrained model cannot be probed for
    timestep/latent sensitivity: every conditioned branch starts at zero.
    """
    torch.manual_seed(0)
    net = NCSNpp(small_config).eval()
    x = torch.randn(1, small_config.num_channels,
                    small_config.image_size, small_config.image_size)
    with torch.no_grad():
        out = net(x, torch.zeros(1, dtype=torch.int64), torch.randn(1, small_config.nz))
    assert out.abs().max() < 1e-3


def test_timestep_embedding_separates_timesteps(small_config):
    """The conditioning pathway itself: distinct t must give distinct embeddings."""
    from backbones.layers import get_timestep_embedding

    emb = get_timestep_embedding(torch.arange(small_config.num_timesteps),
                                 small_config.num_channels_dae)
    assert emb.shape == (small_config.num_timesteps, small_config.num_channels_dae)
    assert torch.isfinite(emb).all()
    for i in range(emb.shape[0]):
        for j in range(i + 1, emb.shape[0]):
            assert not torch.allclose(emb[i], emb[j]), "t={} and t={} collide".format(i, j)


def test_z_transform_responds_to_the_latent(small_config):
    """The latent mapping network must not collapse different z to one code."""
    torch.manual_seed(0)
    net = NCSNpp(small_config).eval()
    with torch.no_grad():
        first = net.z_transform(torch.randn(1, small_config.nz))
        second = net.z_transform(torch.randn(1, small_config.nz))
    assert first.shape == (1, small_config.z_emb_dim)
    assert not torch.allclose(first, second)


def test_ncsnpp_can_reduce_a_reconstruction_loss(small_config):
    """A few optimiser steps must move the loss: catches a dead training path."""
    torch.manual_seed(0)
    net = NCSNpp(small_config)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(1, small_config.num_channels,
                    small_config.image_size, small_config.image_size)
    t = torch.zeros(1, dtype=torch.int64)
    z = torch.randn(1, small_config.nz)
    target = torch.full_like(x, 0.5)

    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(x, t, z), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "loss did not move: {} -> {}".format(losses[0], losses[-1])


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 6, 7, 8])
def test_discriminator_large_accepts_any_batch_size(batch):
    """The minibatch-stddev grouping must divide the batch, whatever its size."""
    net = Discriminator_large(nc=2, ngf=8, t_emb_dim=16, act=nn.LeakyReLU(0.2))
    out = net(torch.randn(batch, 1, 128, 128),
              torch.randint(0, 4, (batch,)),
              torch.randn(batch, 1, 128, 128))
    assert out.shape == (batch, 1)


@pytest.mark.parametrize("batch", [1, 3, 5, 6])
def test_discriminator_small_accepts_any_batch_size(batch):
    net = Discriminator_small(nc=2, ngf=8, t_emb_dim=16, act=nn.LeakyReLU(0.2))
    out = net(torch.randn(batch, 1, 32, 32),
              torch.randint(0, 4, (batch,)),
              torch.randn(batch, 1, 32, 32))
    assert out.shape == (batch, 1)


def test_discriminator_large_backward():
    net = Discriminator_large(nc=2, ngf=8, t_emb_dim=16, act=nn.LeakyReLU(0.2))
    out = net(torch.randn(2, 1, 128, 128), torch.randint(0, 4, (2,)),
              torch.randn(2, 1, 128, 128))
    out.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in net.parameters())


def test_define_g_is_not_wrapped_in_a_parallel_module():
    """init_net must leave parallelism to DDP; a nested wrapper renames keys."""
    net = generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[])
    assert not isinstance(net, nn.DataParallel)
    assert all(not k.startswith('module.') for k in net.state_dict())


def test_translation_networks_preserve_spatial_shape():
    gen = generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[])
    x = torch.randn(2, 1, 64, 64)
    out = gen(x)
    assert out.shape == x.shape

    disc = generator_resnet.define_D(gpu_ids=[])
    # PatchGAN returns a map of patch scores rather than one scalar
    assert disc(out).ndim == 4


@pytest.mark.parametrize("ngf", [16, 32, 64])
def test_define_g_honours_the_channel_width(ngf):
    """--ngf must reach the translation networks, not just the discriminators."""
    net = generator_resnet.define_G(netG='resnet_6blocks', ngf=ngf, gpu_ids=[])
    # the first conv after the reflection pad produces ngf feature maps
    first_conv = [m for m in net.modules() if isinstance(m, nn.Conv2d)][0]
    assert first_conv.out_channels == ngf


def test_define_d_honours_the_channel_width():
    net = generator_resnet.define_D(ndf=16, gpu_ids=[])
    first_conv = [m for m in net.modules() if isinstance(m, nn.Conv2d)][0]
    assert first_conv.out_channels == 16
