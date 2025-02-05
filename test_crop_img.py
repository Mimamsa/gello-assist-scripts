"""Test image cropping function and display cropped image.

The image is captured from the capture card (Elgato Game Capture HD60 X) connected to GoPro Hero 12. The GoPro image aspact ratio is set to 1.33. The capture card zero pads images to 1080p and then resize it to a requested supported resolution. Supported resolutions for Elgato Game Capture HD please see the link: https://help.elgato.com/hc/en-us/articles/360027952992-Supported-resolutions-for-Elgato-Game-Capture-HD

This script cropped out the zero paddings areas of images in order to change aspect ratio from 1.78 to 1.33.

"""
import click
import cv2
import pickle

from utils.img_proc import crop_to_4w3h

INPUT_FILE = ''

@click.command()
@click.option('-i', '--input_file', default=INPUT_FILE, help='Pickle file of a single frame.')
def main(input_file):

    with open(input_file, "rb") as f:
        obs = pickle.load(f)

    print('joint_positions: ', obs["joint_positions"].shape)
    print('joint_velocities: ', obs["joint_velocities"].shape)
    print('ee_pose: ', obs["ee_pose"])
    print('gripper_position: ', obs["gripper_position"].shape)
    print('control: ', obs['control'].shape)
    print('wrist_rgb shape: ', obs['wrist_rgb'].shape)
    img = obs['wrist_rgb']

    # Crop image from (1920, 1080) to 4:3
    img = crop_to_4w3h(img)

    cv2.imshow('Image', img[:,:,::-1])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()
