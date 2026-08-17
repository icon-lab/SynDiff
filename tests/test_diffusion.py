"""Properties the forward/reverse diffusion coefficients have to satisfy."""
import numpy as np
import pytest
import torch

from diffusion import (Diffusion_Coefficients, Posterior_Coefficients, extract,
                       get_sigma_schedule, get_time_schedule, q_sample_pairs,
                       sample_posterior)


def test_sigma_schedule_shapes_and_bounds(small_config):
    sigmas, a_s, betas = get_sigma_schedule(small_config, device='cpu')
    n = small_config.num_timesteps
    assert sigmas.shape == a_s.shape == betas.shape == (n + 1,)
    assert torch.all(betas >= 0) and torch.all(betas <= 1)
    # a_s = sqrt(1 - beta) is the per-step signal retention
    assert torch.allclose(a_s, torch.sqrt(1 - betas), atol=1e-6)
    assert torch.allclose(sigmas, betas ** 0.5, atol=1e-6)


def test_variance_preserving_identity(small_config):
    """a_s_cum^2 + sigmas_cum^2 == 1 -- the VP property q_sample relies on."""
    coeff = Diffusion_Coefficients(small_config, device='cpu')
    total = coeff.a_s_cum ** 2 + coeff.sigmas_cum ** 2
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_cumulative_signal_decays(small_config):
    coeff = Diffusion_Coefficients(small_config, device='cpu')
    a_s_cum = coeff.a_s_cum
    assert torch.all(a_s_cum[1:] <= a_s_cum[:-1] + 1e-6), "signal must not grow with t"
    assert torch.all(coeff.sigmas_cum[1:] >= coeff.sigmas_cum[:-1] - 1e-6)


def test_posterior_variance_is_positive(small_config):
    pos = Posterior_Coefficients(small_config, device='cpu')
    assert torch.all(pos.posterior_variance >= 0)
    assert torch.all(torch.isfinite(pos.posterior_log_variance_clipped))
    assert torch.all(torch.isfinite(pos.posterior_mean_coef1))
    assert torch.all(torch.isfinite(pos.posterior_mean_coef2))


def test_posterior_collapses_to_x0_at_t_zero(small_config):
    """At t == 0 the posterior mean is exactly x_0: coef1 == 1, coef2 == 0."""
    pos = Posterior_Coefficients(small_config, device='cpu')
    assert pos.posterior_mean_coef1[0] == pytest.approx(1.0, abs=1e-4)
    assert pos.posterior_mean_coef2[0] == pytest.approx(0.0, abs=1e-4)


def test_posterior_mean_coefficients_are_non_negative(small_config):
    pos = Posterior_Coefficients(small_config, device='cpu')
    assert torch.all(pos.posterior_mean_coef1 >= 0)
    assert torch.all(pos.posterior_mean_coef2 >= 0)
    # weight on x_0 falls off as the step index grows
    c1 = pos.posterior_mean_coef1
    assert torch.all(c1[1:] <= c1[:-1] + 1e-6)


def test_posterior_variance_never_exceeds_beta(small_config):
    pos = Posterior_Coefficients(small_config, device='cpu')
    assert torch.all(pos.posterior_variance <= pos.betas + 1e-6)


def test_reciprocal_alpha_helpers_are_consistent(small_config):
    pos = Posterior_Coefficients(small_config, device='cpu')
    product = pos.sqrt_recip_alphas_cumprod * pos.sqrt_alphas_cumprod
    assert torch.allclose(product, torch.ones_like(product), atol=1e-5)


def test_time_schedule_is_increasing_and_bounded(small_config):
    T = get_time_schedule(small_config, device='cpu')
    assert T.shape == (small_config.num_timesteps + 1,)
    assert torch.all(T[1:] > T[:-1])
    assert T[0] > 0 and T[-1] <= 1.0


def test_extract_selects_per_sample_coefficients():
    values = torch.tensor([10., 20., 30.])
    t = torch.tensor([2, 0])
    out = extract(values, t, (2, 1, 4, 4))
    assert out.shape == (2, 1, 1, 1)
    assert out.flatten().tolist() == [30., 10.]


def test_q_sample_pairs_keeps_shape_and_is_noisier_at_t_plus_one(small_config):
    coeff = Diffusion_Coefficients(small_config, device='cpu')
    torch.manual_seed(0)
    x0 = torch.randn(4, 1, 8, 8)
    t = torch.zeros(4, dtype=torch.int64)
    x_t, x_tp1 = q_sample_pairs(coeff, x0, t)
    assert x_t.shape == x_tp1.shape == x0.shape
    # at t=0 the pair is one diffusion step apart, so x_tp1 is further from x0
    assert (x_tp1 - x0).abs().mean() > (x_t - x0).abs().mean()


def test_sample_posterior_is_deterministic_at_t_zero(small_config):
    """The nonzero_mask must suppress the noise term for t == 0."""
    pos = Posterior_Coefficients(small_config, device='cpu')
    x0 = torch.randn(3, 1, 8, 8)
    xt = torch.randn(3, 1, 8, 8)
    t = torch.zeros(3, dtype=torch.int64)
    first = sample_posterior(pos, x0, xt, t)
    second = sample_posterior(pos, x0, xt, t)
    assert torch.allclose(first, second)


def test_sample_posterior_is_stochastic_for_positive_t(small_config):
    pos = Posterior_Coefficients(small_config, device='cpu')
    x0 = torch.randn(3, 1, 8, 8)
    xt = torch.randn(3, 1, 8, 8)
    t = torch.full((3,), small_config.num_timesteps - 1, dtype=torch.int64)
    assert not torch.allclose(sample_posterior(pos, x0, xt, t),
                              sample_posterior(pos, x0, xt, t))
