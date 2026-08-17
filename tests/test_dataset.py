"""Loading, padding and normalisation of the .mat volumes."""
import h5py
import numpy as np
import pytest
import torch

from dataset import CreateDatasetSynthesis, LoadDataSet


def write_mat(path, array, variable='data_fs'):
    with h5py.File(path, 'w') as f:
        f.create_dataset(variable, data=array)
    return str(path)


def test_loads_and_normalises_to_minus_one_one(tmp_path):
    raw = np.linspace(0, 1, 2 * 152 * 256, dtype=np.float32).reshape(2, 152, 256)
    data = LoadDataSet(write_mat(tmp_path / 'a.mat', raw))
    assert data.shape == (2, 1, 256, 256)
    assert data.dtype == np.float32
    # (x - 0.5) / 0.5 maps [0, 1] onto [-1, 1]
    assert data.min() >= -1.0 - 1e-6 and data.max() <= 1.0 + 1e-6


def test_padding_can_be_disabled(tmp_path):
    raw = np.zeros((2, 152, 256), dtype=np.float32)
    data = LoadDataSet(write_mat(tmp_path / 'a.mat', raw), padding=False)
    assert data.shape == (2, 1, 256, 152)


@pytest.mark.parametrize("width", [151, 152, 153, 255, 256])
def test_padding_always_reaches_the_target_size(tmp_path, width):
    """An odd width used to lose a pixel to integer truncation."""
    raw = np.zeros((2, width, 256), dtype=np.float32)
    data = LoadDataSet(write_mat(tmp_path / 'a.mat', raw))
    assert data.shape[2:] == (256, 256)


def test_padding_keeps_the_image_centred(tmp_path):
    raw = np.ones((1, 152, 256), dtype=np.float32)
    data = LoadDataSet(write_mat(tmp_path / 'a.mat', raw), Norm=False)
    column_sums = data[0, 0].sum(axis=0)
    filled = np.nonzero(column_sums)[0]
    assert filled.size == 152
    # at most one pixel of asymmetry, and it goes to the far side
    assert abs(filled[0] - (256 - 1 - filled[-1])) <= 1


def test_oversized_input_is_rejected_with_a_useful_message(tmp_path):
    raw = np.zeros((2, 300, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="does not fit"):
        LoadDataSet(write_mat(tmp_path / 'a.mat', raw))


def test_missing_file_names_the_expected_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="data_<phase>_<contrast>"):
        LoadDataSet(str(tmp_path / 'nope.mat'))


def test_wrong_variable_lists_what_the_file_holds(tmp_path):
    path = write_mat(tmp_path / 'a.mat', np.zeros((2, 152, 256), np.float32),
                     variable='images')
    with pytest.raises(KeyError, match="images"):
        LoadDataSet(path)


def test_unexpected_dimensionality_is_rejected(tmp_path):
    path = write_mat(tmp_path / 'a.mat', np.zeros((2, 152), np.float32))
    with pytest.raises(ValueError, match="dimensions"):
        LoadDataSet(path)


def test_create_dataset_pairs_the_two_contrasts(tmp_path):
    for contrast in ('T1', 'T2'):
        write_mat(tmp_path / 'data_train_{}.mat'.format(contrast),
                  np.zeros((3, 152, 256), np.float32))
    dataset = CreateDatasetSynthesis('train', str(tmp_path), 'T1', 'T2')
    assert len(dataset) == 3
    first, second = dataset[0]
    assert isinstance(first, torch.Tensor) and first.shape == (1, 256, 256)
    assert second.shape == (1, 256, 256)


def test_mismatched_slice_counts_are_rejected(tmp_path):
    write_mat(tmp_path / 'data_train_T1.mat', np.zeros((4, 152, 256), np.float32))
    write_mat(tmp_path / 'data_train_T2.mat', np.zeros((3, 152, 256), np.float32))
    with pytest.raises(ValueError, match="different number of slices"):
        CreateDatasetSynthesis('train', str(tmp_path), 'T1', 'T2')


@pytest.mark.parametrize("image_size", [128, 192, 256])
def test_target_size_is_configurable(tmp_path, image_size):
    """--image_size has to reach the loader, not just the network."""
    raw = np.zeros((2, 100, 120), dtype=np.float32)
    data = LoadDataSet(write_mat(tmp_path / 'a.mat', raw), target_size=image_size)
    assert data.shape == (2, 1, image_size, image_size)


def test_create_dataset_forwards_the_image_size(tmp_path):
    for contrast in ('T1', 'T2'):
        write_mat(tmp_path / 'data_train_{}.mat'.format(contrast),
                  np.zeros((2, 100, 120), np.float32))
    dataset = CreateDatasetSynthesis('train', str(tmp_path), 'T1', 'T2', image_size=128)
    first, second = dataset[0]
    assert first.shape == second.shape == (1, 128, 128)


def test_input_larger_than_the_target_is_rejected(tmp_path):
    raw = np.zeros((2, 200, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="does not fit"):
        LoadDataSet(write_mat(tmp_path / 'a.mat', raw), target_size=128)
