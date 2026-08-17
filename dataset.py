import os

import torch.utils.data
import numpy as np, h5py
import random


def CreateDatasetSynthesis(phase, input_path, contrast1 = 'T1', contrast2 = 'T2',
                           image_size = 256):

    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast1)
    data_fs_s1=LoadDataSet(target_file, target_size=image_size)

    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast2)
    data_fs_s2=LoadDataSet(target_file, target_size=image_size)

    if data_fs_s1.shape[0] != data_fs_s2.shape[0]:
        raise ValueError(
            "'{}' and '{}' hold a different number of slices ({} vs {}); the two "
            "contrasts must be aligned slice by slice.".format(
                contrast1, contrast2, data_fs_s1.shape[0], data_fs_s2.shape[0]))

    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1),torch.from_numpy(data_fs_s2))
    return dataset



#Dataset loading from load_dir, zero-padded out to target_size squared
def LoadDataSet(load_dir, variable = 'data_fs', padding = True, Norm = True,
                target_size = 256):
    if not os.path.isfile(load_dir):
        raise FileNotFoundError(
            "No such data file: '{}'. Files are expected to be named "
            "data_<phase>_<contrast>.mat inside --input_path.".format(load_dir))

    with h5py.File(load_dir,'r') as f:
        if variable not in f:
            raise KeyError(
                "'{}' does not contain a '{}' variable (found: {}).".format(
                    load_dir, variable, ', '.join(f.keys()) or 'nothing'))
        raw = np.array(f[variable])

    if raw.ndim==3:
        data=np.expand_dims(np.transpose(raw,(0,2,1)),axis=1)
    elif raw.ndim==4:
        data=np.transpose(raw,(1,0,3,2))
    else:
        raise ValueError(
            "'{}' has {} dimensions; expected 3 (slices, width, height) or 4.".format(
                load_dir, raw.ndim))
    data=data.astype(np.float32)
    if padding:
        pads = []
        for axis in (2, 3):
            size = data.shape[axis]
            if size > target_size:
                raise ValueError(
                    "'{}' has a {}x{} image size, which does not fit the {}x{} grid "
                    "SynDiff pads to. Crop or resample the volume first.".format(
                        load_dir, data.shape[2], data.shape[3], target_size, target_size))
            # split the padding across both sides; the extra pixel of an odd
            # difference goes to the far side so the result is exactly target_size
            before = (target_size - size) // 2
            pads.append((before, target_size - size - before))
        print('padding in x-y with:'+str(pads[0])+'-'+str(pads[1]))
        data=np.pad(data,((0,0),(0,0),pads[0],pads[1]))
    if Norm:
        data=(data-0.5)/0.5
    return data
