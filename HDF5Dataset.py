import numpy as np
import h5py
import torch
import torch.utils.data as data

class HDF5Dataset(data.Dataset):

    def __init__(self, filename):
        self.h5file = h5py.File(filename, 'r')
        self.length = self.h5file['train_camera_data'].shape[0]

    def __getitem__(self, index):
        camera_data = self.h5file['train_camera_data'][index, :, :, :]
        metadata = self.h5file['train_metadata'][index, :, :, :]
        target_data = self.h5file['train_target_data'][index, :]
        return torch.from_numpy(camera_data).float(), torch.from_numpy(metadata).float(), torch.from_numpy(target_data).float()


    def __len__(self):
        return self.length
