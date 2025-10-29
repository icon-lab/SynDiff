import h5py
import torch
import pandas as pd
import json
import operator
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random


# --- Filtering ---
def dict_to_filter_fn(filter_dict):
    ops = {
        'eq': operator.eq,
        'ne': operator.ne,
        'lt': operator.lt,
        'le': operator.le,
        'gt': operator.gt,
        'ge': operator.ge,
        'in': lambda a, b: a.isin(b),
        'range': lambda a, b: handle_range(a, b)
    }

    def parse_key_op(key):
        if '__' in key:
            field, op = key.split('__')
            return field, ops[op]
        return key, operator.eq

    def handle_range(series, value):
        if isinstance(value, str) and ':' in value:
            start, end = map(int, value.split(':'))
            return (series >= start) & (series <= end)
        if isinstance(value, list) and all(isinstance(v, int) for v in value):
            return series.isin(value)
        raise ValueError(f"Unsupported format for range filter: {value}")

    def filter_fn(df):
        mask = pd.Series([True] * len(df), index=df.index)
        for key, value in filter_dict.items():
            k, op = parse_key_op(key)
            mask &= op(df[k], value)
        return mask

    return filter_fn


# --- Albumentations Transform Loader ---
class PerImageMinMaxNormalize(A.ImageOnlyTransform):
    """Normalizes an image using its own min and max values."""
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, image, **params):
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val < 1e-6:
            return image.astype(np.float32) * 0.0
        return (image - min_val) / (max_val - min_val)

# --- Group-based Normalization Transform ---
class GroupMinMaxNormalize(A.BasicTransform):
    """
    Normalizes an image using pre-computed min/max values for a group.
    The group is identified by keys in the image's metadata.
    
    This transform inherits from BasicTransform to gain full control over the
    input data dictionary via the __call__ method, which is necessary to
    access metadata alongside the image.
    """
    def __init__(self, stats_path, contrast_key='contrast', patient_id_key='patient_id', image_type_key='image_type', always_apply=True, p=1.0):
        """
        Args:
            stats_path (str): Path to the JSON file with normalization stats.
            contrast_key (str): The key for 'contrast' in the metadata dictionary.
            patient_id_key (str): The key for 'patient_id' in the metadata dictionary.
            image_type_key (str): The key for 'image_type' in the metadata dictionary.
        """
        super(GroupMinMaxNormalize, self).__init__(p=p)
        
        self.contrast_key = contrast_key
        self.patient_id_key = patient_id_key
        self.image_type_key = image_type_key
        
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
        print(f"Loaded normalization stats for {len(self.stats)} contrasts.")

    def __call__(self, force_apply=False, **data):
        metadata = data.get('metadata')
        if metadata is None:
            raise ValueError("GroupMinMaxNormalize requires 'metadata' to be passed to the transform.")

        image = data['image']

        try:
            # Look up the correct stats using keys from the metadata
            contrast = metadata[self.contrast_key]
            patient_id = metadata[self.patient_id_key]
            image_type = metadata[self.image_type_key]

            group_stats = self.stats[contrast][image_type][patient_id]
            min_val = group_stats['min']
            max_val = group_stats['max']
        except KeyError as e:
            print(f"Metadata causing error: {metadata}")
            raise KeyError(f"Could not find stats for the given metadata. Missing key: {e}")

        # Avoid division by zero
        if max_val - min_val < 1e-6:
            normalized_image = image.astype(np.float32) * 0.0
        else:
            # Ensure image is float for the division and perform normalization
            image = image.astype(np.float32)
            normalized_image = (image - min_val) / (max_val - min_val)
        
        data['image'] = normalized_image
        return data

    @property
    def targets(self):
        # We are overriding __call__, so we don't need the default target mechanism.
        # Returning an empty dict is the safest approach.
        return {}

    def get_transform_init_args_names(self):
        return ("stats_path", "contrast_key", "patient_id_key", "image_type_key")


def build_albumentations_transform(config, stage="fit"):
    transforms = []

    custom_transforms = {
        "PerImageMinMaxNormalize": PerImageMinMaxNormalize,
        "GroupMinMaxNormalize": GroupMinMaxNormalize,
    }

    for t in config.get(stage, []):
        for name, params in t.items():
            if name in custom_transforms:
                cls = custom_transforms[name]
            else:
                cls = getattr(A, name)
            transforms.append(cls(**params) if params else cls())

    transforms.append(ToTensorV2(transpose_mask=True))

    return A.Compose(transforms)


# --- Dataset Class ---
class HDF5ContrastDataset(Dataset):
    def __init__(self, hdf5_path, filter, transform=None, stage="fit", split=None):
        self.hdf5_path = hdf5_path
        self.stage = stage
        self.split = split

        if isinstance(filter, str):
            with open(filter, 'r') as f:
                self.filter_dict = json.load(f)
        elif isinstance(filter, dict):
            self.filter_dict = filter
        else:
            raise ValueError("Filter must be a dict or path to a JSON file.")

        filter_fn = dict_to_filter_fn(self.filter_dict)

        metadata = pd.read_hdf(hdf5_path, key="metadata")
        with h5py.File(hdf5_path, 'r') as h5f:
            if self.split is not None:
                if 'split' not in metadata.columns:
                    raise ValueError("'split' column not found in metadata.")
                metadata = metadata[metadata['split'] == self.split]

            filtered_df = metadata[filter_fn(metadata)].copy()
            indices = filtered_df['index'].values
            self.images = h5f['images'][indices]
            self.masks = h5f['masks'][indices]
            self.metadata = filtered_df.reset_index(drop=True)

        if transform is None:
            self.transform = None
        else:
            if isinstance(transform, str):
                with open(transform, 'r') as f:
                    transform = json.load(f)
            self.transform = build_albumentations_transform(transform, stage)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        metadata_row = self.metadata.iloc[idx].to_dict()

        if image.ndim == 2:
            image = image[..., None]
        if mask.ndim == 2:
            mask = mask[..., None]

        if self.transform:
            # This call is correct. The fix is inside the GroupMinMaxNormalize class.
            augmented = self.transform(image=image, mask=mask, metadata=metadata_row)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
            mask = torch.tensor(mask.transpose(2, 0, 1), dtype=torch.float32)

        if torch.isnan(image).any() or torch.isinf(image).any():
            print(f"NaN or Inf in image at idx {idx}. Metadata: {metadata_row}")
            image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(mask).any() or torch.isinf(mask).any():
            print(f"NaN or Inf in mask at idx {idx}. Metadata: {metadata_row}")
            mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "image": image,
            "mask": mask,
            "metadata": metadata_row
        }

    def get_metadata(self):
        return self.metadata

# --- Paired Dataset Class ---
class PairedHDF5ContrastDataset(Dataset):
    def __init__(self, hdf5_path_template, hdf5_path_moving,
                 filter_template, filter_moving,
                 transform_template=None, transform_moving=None,
                 stage="fit", split=None):
        self.template_ds = HDF5ContrastDataset(
            hdf5_path=hdf5_path_template,
            filter=filter_template,
            transform=transform_template,
            stage=stage,
            split=split
        )
        self.moving_ds = HDF5ContrastDataset(
            hdf5_path=hdf5_path_moving,
            filter=filter_moving,
            transform=transform_moving,
            stage=stage,
            split=split
        )

        meta_template = self.template_ds.get_metadata()
        meta_moving = self.moving_ds.get_metadata()

        merged = pd.merge(
            meta_template,
            meta_moving,
            on=["patient_id", "z_dim"],
            suffixes=("_template", "_moving")
        )

        self.matched_indices = []
        for _, row in merged.iterrows():
            idx_template = meta_template[
                (meta_template["patient_id"] == row["patient_id"]) &
                (meta_template["z_dim"] == row["z_dim"])
            ].index[0]

            idx_moving = meta_moving[
                (meta_moving["patient_id"] == row["patient_id"]) &
                (meta_moving["z_dim"] == row["z_dim"])
            ].index[0]

            self.matched_indices.append((idx_template, idx_moving))

    def __len__(self):
        return len(self.matched_indices)

    def __getitem__(self, idx):
        idx_template, idx_moving = self.matched_indices[idx]

        sample_template = self.template_ds[idx_template]
        sample_moving = self.moving_ds[idx_moving]

        return {
            "image_template": sample_template["image"],
            "mask_template": sample_template["mask"],
            "metadata_template": sample_template["metadata"],

            "image_moving": sample_moving["image"],
            "mask_moving": sample_moving["mask"],
            "metadata_moving": sample_moving["metadata"]
        }
