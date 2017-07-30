"""Data preprocessing code for bddmodelcar."""
import traceback
import logging

from Parameters import ARGS
import Data
import Batch
import Utils

import matplotlib.pyplot as plt

from nets.SqueezeNet import SqueezeNet
import torch
import h5py


def main():
    ARGS.batch_size = 1
    # Set Up PyTorch Environment
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.set_device(ARGS.gpu)
    torch.cuda.device(ARGS.gpu)

    net = SqueezeNet().cuda()

    data = Data.Data()
    batch = Batch.Batch(net)
    rate_counter = Utils.RateCounter()

    h5File = h5py.File("/data/tpankaj/preprocess.hdf5", "w")
    train_len = len(data.train_index.valid_data_moments)
    val_len = len(data.val_index.valid_data_moments)
    train_camera_data = h5File.create_dataset("train_camera_data", (train_len, 12, 94, 168), chunks=(ARGS.batch_size, 12, 94, 168), dtype='float32', compression='lzf')
    train_metadata = h5File.create_dataset("train_metadata", (train_len, 128, 23, 41), chunks=(ARGS.batch_size, 128, 23, 41), dtype='float32', compression='lzf')
    train_target_data = h5File.create_dataset("train_target_data", (train_len, 20), chunks=(ARGS.batch_size, 20), dtype='float32', compression='lzf')

    # Save training data
    while not data.train_index.epoch_complete:  # Epoch of training
        camera_data, metadata, target_data = batch.fill(data, data.train_index)
        #print("Start: " + str(data.train_index.ctr - ARGS.batch_size))
        #print("End: " + str(data.train_index.ctr))
        train_camera_data[data.train_index.ctr - ARGS.batch_size : data.train_index.ctr, :, :, :] = camera_data.cpu().numpy()
        train_metadata[data.train_index.ctr - ARGS.batch_size : data.train_index.ctr, :, :, :] = metadata.cpu().numpy()
        train_target_data[data.train_index.ctr - ARGS.batch_size : data.train_index.ctr, :] = target_data.cpu().numpy()
        if (data.train_index.ctr % 10000) == 0:
            print(data.train_index.ctr)
        rate_counter.step()

if __name__ == '__main__':
    main()
