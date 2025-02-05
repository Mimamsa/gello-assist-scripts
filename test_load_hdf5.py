""" Loads demonstration data from a HDF5 file.

"""
import click
import cv2
import numpy as np

from utils.data_io import load_hdf5_demo
from utils.viz import display_tcp_pos

DEMO_FILE = 'D://work/datasets/cup-pnp/gello/episode_0.hdf5'


@click.command()
@click.option('-i', '--input_file', default=DEMO_FILE, help='HDF5 file of a demonstration.')
def main(input_file):

    data = load_hdf5_demo(input_file)
    for k in data.keys():
        print(k, data[k].shape)

    # visualize
    for img in data['/observations/images/wrist_rgb']:
        cv2.imshow('Image', img[:,:,::-1])
        cv2.waitKey(50)

    display_tcp_pos(data['/observations/ee_pose'], fps=20)


if __name__=='__main__':
    main()