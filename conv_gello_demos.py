"""

"""
import click
import os
import cv2
import numpy as np
from natsort import natsorted
#from datetime import datetime
from tqdm import tqdm
from utils.data_io import load_pkl_demo, save_hdf5_demo


INPUT_DIR = '/media/hungyi/Data_sdb5/home/gello_dataset/gello_pnp_cup/gello'
OUTPUT_DIR = '/media/hungyi/Data_sdb5/home/gello_dataset/gello_pnp_cup/converted'


def batch_resize(imgs, size=(640,480)):
    """
    """
    ret = []
    for img in imgs:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        ret.append(img)
    return np.stack(ret)


@click.command()
@click.option('-i', '--input_dir', default=INPUT_DIR, help='Root directory of a GELLO demonstrations.')
@click.option('-o', '--output_dir', default=OUTPUT_DIR, help='Directory contains saved demonstration files.')
def main(input_dir, output_dir):

    dirs = natsorted(os.listdir(input_dir))
    camera_names=['wrist_rgb']

    for dname in tqdm(dirs):
        #dt = datetime.strptime(dname, '%m%d_%H%M%S').replace(year=2025)
        demo_dir = os.path.join(input_dir, dname)
        data = load_pkl_demo(demo_dir)

        # Process image size
        for cam_name in camera_names:
            data[cam_name] = batch_resize(data[cam_name])

        save_file = os.path.join(output_dir, f'{dname}.hdf5')
        save_hdf5_demo(save_file, data, camera_names)


if __name__=='__main__':
    main()
