import os
import h5py
from Parameters import ARGS


def main():
    hdf5_path = ARGS.data_path + '/hdf5/runs/'
    os.chdir(hdf5_path)
    for hdf5_file in os.listdir(hdf5_path):
        pass


if __name == '__main__':
    main()
