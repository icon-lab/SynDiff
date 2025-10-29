import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import DataLoader
from pytorch_dataset import HDF5ContrastDataset

def save_dataset_as_mat(dataset, out_file, var='data_fs', compression='gzip'):
    """Save so that LoadDataSet() shows the image upright."""

    imgs = []
    for i in range(len(dataset)):
        img = dataset[i]['image']          # (1,256,256)  channel‑first
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = img.squeeze(0) .T            # <-- transpose here (W, H)
        imgs.append(img.astype(np.float32))

    imgs = np.stack(imgs, axis=0)          # (N,256,256)
    with h5py.File(out_file, 'w') as f:
        f.create_dataset(var, data=imgs,
                         dtype='float32', compression=compression)

    print(f'✔ Saved {len(imgs)} slices to {out_file} (will load as (N,1,256,256))')

def get_patient_z_dim_combinations(overview_df_filtered, contrast_list):
    contrast_groups = overview_df_filtered[overview_df_filtered["contrast"].isin(contrast_list)].groupby('contrast').apply(
        lambda g: set(zip(g['patient_id'], g['z_dim']))
    )

    # Step 2: Take the intersection of all sets
    common_combinations = set.intersection(*contrast_groups)

    # Step 3: Convert back to a DataFrame if needed
    result_df = pd.DataFrame(list(common_combinations), columns=['patient_id', 'z_dim'])

    return result_df

def load_dataset(base_path, contrast, image_type, cfg_transform, split="train", image_dim=0):
    # Load the synthetic dataset
    cfg_filters = {
        "contrast__in": [contrast],
        "non_zero": True,
        "image_dim": 0,
        "image_type": image_type,
        "split":split,
    }
    dataset = HDF5ContrastDataset(
        hdf5_path=f"{base_path}/data_{contrast}.h5",
        filter=cfg_filters,
        transform=cfg_transform,
        stage="eval",  
    )
    return dataset

def save_both_dataset_as_mat(dataset1, dataset2, out_file1, out_file2, var='data_fs', compression='gzip'):
    """Save so that LoadDataSet() shows the image upright."""

    imgs_1 = []
    imgs_2 = []
    already_vistied_j = set()  # To avoid duplicate processing of dataset2
    for i in range(len(dataset1)):
        img_1 = dataset1[i]['image']          # (1,256,256)  channel‑first
        img_1_metadata = dataset1[i]['metadata']
        for j in range(len(dataset2)):
            if j in already_vistied_j:
                continue
            img_2_metadata = dataset2[j]['metadata']
            if img_1_metadata['patient_id'] == img_2_metadata['patient_id'] and img_1_metadata['z_dim'] == img_2_metadata['z_dim']:
                img_2 = dataset2[j]['image']
                if isinstance(img_1, torch.Tensor):
                    img_1 = img_1.cpu().numpy()
                img_1 = img_1.squeeze(0).T  # <-- transpose here (W, H)
                imgs_1.append(img_1.astype(np.float32))
                if isinstance(img_2, torch.Tensor):
                    img_2 = img_2.cpu().numpy()
                img_2 = img_2.squeeze(0).T # <-- transpose here
                imgs_2.append(img_2.astype(np.float32))
                already_vistied_j.add(j)
                break

    imgs_1 = np.stack(imgs_1, axis=0) # (N,256,256)
    with h5py.File(out_file1, 'w') as f:
        f.create_dataset(var, data=imgs_1,
                         dtype='float32', compression=compression)
        
    print(f'Saved {len(imgs_1)} slices to {out_file1} (will load as (N,1,256,256))')
        
    imgs_2 = np.stack(imgs_2, axis=0) # (N,256,256)
    with h5py.File(out_file2, 'w') as f:
        f.create_dataset(var, data=imgs_2,
                         dtype='float32', compression=compression)

    print(f'Saved {len(imgs_2)} slices to {out_file2} (will load as (N,1,256,256))')


def create_datasets(contrast1, contrast2, image_type1, image_type2, cfg_transform, base_path, output_path):

    dataset1_train = load_dataset(base_path, contrast1, image_type1, cfg_transform, split="train")
    dataset2_train = load_dataset(base_path, contrast2, image_type2, cfg_transform, split="train")

    dataset1_val = load_dataset(base_path, contrast1, image_type1, cfg_transform, split="val")
    dataset2_val = load_dataset(base_path, contrast2, image_type2, cfg_transform, split="val")

    dataset1_test = load_dataset(base_path, contrast1, image_type1, cfg_transform, split="test")
    dataset2_test = load_dataset(base_path, contrast2, image_type2, cfg_transform, split="test")

    out_file_1_train = f"{output_path}/{contrast1}_{contrast2}_train.h5"
    out_file_2_train = f"{output_path}/{contrast2}_train.h5"

    out_file_1_val = f"{output_path}/{contrast1}_{contrast2}_val.h5"
    out_file_2_val = f"{output_path}/{contrast2}_val.h5"

    out_file_1_test = f"{output_path}/{contrast1}_{contrast2}_test.h5"
    out_file_2_test = f"{output_path}/{contrast2}_test.h5"

    save_both_dataset_as_mat(dataset1_train, dataset2_train, out_file_1_train, out_file_2_train, var='data_fs', compression='gzip')
    save_both_dataset_as_mat(dataset1_val, dataset2_val, out_file_1_val, out_file_2_val, var='data_fs', compression='gzip')
    save_both_dataset_as_mat(dataset1_test, dataset2_test, out_file_1_test, out_file_2_test, var='data_fs', compression='gzip')


def main():
    cfg_transform = {
    "eval": [
        {
        "GroupMinMaxNormalize": {
            "stats_path": "/home/students/studweilc1/SynthRegGAN/data/minmax_values.json"
        }
        },

    ]
    }

    base_path = "/home/students/studweilc1/SynthRegGAN/data"
    output_path = "/home/students/studweilc1/SynDiff/data/my_data_group"

    contrast1 = "DIXON"
    contrast2 = "BOLD"
    image_type1 = "W"
    image_type2 = "s"

    create_datasets(contrast1, contrast2, image_type1, image_type2, cfg_transform, base_path, output_path)

if __name__ == "__main__":
    main()