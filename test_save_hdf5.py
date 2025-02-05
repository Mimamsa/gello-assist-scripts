"""
Loads demonstration data from pickle files in a directory and save as a HDF5 file.
"""
import os
import click
import cv2
import numpy as np

from utils.viz import display_tcp_pos
from utils.data_io import load_pkl_demo, save_hdf5_demo


DEMO_DIR = 'D://work/datasets/cup-pnp/gello/0123_160558'
OUT_DIR = 'D://work/datasets/cup-pnp/gello'


def batch_resize(imgs, size=(640,480)):
    """
    """
    ret = []
    for img in imgs:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        ret.append(img)
    return np.stack(ret)


@click.command()
@click.option('-i', '--input_dir', default=DEMO_DIR, help='Directory of a GELLO demonstration.')
@click.option('-o', '--output_dir', default=OUT_DIR, help='Directory contains saved demonstration file.')
def main(input_dir, output_dir):

    data = load_pkl_demo(input_dir)
    for k in data.keys():
        print(k, data[k].shape)

    # display_tcp_pos(tcp_poses, fps=20)

    # Process image size
    data['wrist_rgb'] = batch_resize(data['wrist_rgb'])

    # visualize
    # for img in data['wrist_rgb']:
        # cv2.imshow('Image', img[:,:,::-1])
        # cv2.waitKey(50)

    max_timesteps = len(data['timestamps'])
    camera_names = ['wrist_rgb']
    episode_idx = 0
    output_file = os.path.join(output_dir, f'episode_{episode_idx}.hdf5')
    save_hdf5_demo(output_file, data, camera_names)


if __name__=='__main__':
    main()