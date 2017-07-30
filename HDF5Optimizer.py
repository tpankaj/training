from __future__ import print_function
import os
import h5py
import numpy as np
from Parameters import ARGS


def main():
    hdf5_path = ARGS.data_path + '/hdf5/runs/'
    os.chdir(hdf5_path)
    for hdf5_file in os.listdir(hdf5_path):
        try:
            print(hdf5_file)
            f = h5py.File(hdf5_file)
            segments = f['segments']
            for run_code in segments:
                run_code_group = segments[run_code]
                for dset_name in run_code_group:
                    dset = run_code_group[dset_name]
                    dset_full_name = dset.name
                    dset_dtype = dset.dtype
                    raw_data = dset[:]
                    del f['segments'][run_code][dset_name]
                    if len(raw_data.shape) == 4:
                        f.create_dataset(
                            dset_full_name,
                            raw_data.shape,
                            chunks=(
                                1,
                                raw_data.shape[1],
                                raw_data.shape[2],
                                raw_data.shape[3]),
                            dtype=dset_dtype)
                    else:
                        f.create_dataset(
                            dset_full_name,
                            raw_data.shape,
                            chunks=True,
                            dtype=dset_dtype)
        except BaseException:
            pass


if __name__ == '__main__':
    main()
