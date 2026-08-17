"""Checkpoints must survive the round trip between parallel wrappers."""
import torch
import torch.nn as nn

from train import load_model_state, strip_module_prefix


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)


def test_strips_no_prefix_from_a_single_process_checkpoint():
    assert list(strip_module_prefix({'conv.weight': 1})) == ['conv.weight']


def test_strips_one_prefix_from_a_ddp_checkpoint():
    assert list(strip_module_prefix({'module.conv.weight': 1})) == ['conv.weight']


def test_strips_both_prefixes_from_a_ddp_over_dataparallel_checkpoint():
    """Revisions that wrapped DataParallel inside DDP wrote two levels."""
    assert list(strip_module_prefix({'module.module.conv.weight': 1})) == ['conv.weight']


def test_prefix_stripping_preserves_values_and_arity():
    state = {'module.a': torch.zeros(2), 'module.b': torch.ones(3)}
    out = strip_module_prefix(state)
    assert set(out) == {'a', 'b'}
    assert torch.equal(out['b'], torch.ones(3))


def test_load_model_state_accepts_every_layout():
    model = Tiny()
    reference = Tiny()
    for prefix in ('', 'module.', 'module.module.'):
        state = {prefix + k: v for k, v in reference.state_dict().items()}
        load_model_state(model, state)
        for key, value in reference.state_dict().items():
            assert torch.equal(model.state_dict()[key], value)


def test_load_model_state_round_trips_through_a_real_save(tmp_path):
    saved = Tiny()
    path = tmp_path / 'ckpt.pth'
    torch.save(saved.state_dict(), path)

    loaded = Tiny()
    load_model_state(loaded, torch.load(path, map_location='cpu'))
    for key, value in saved.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)
