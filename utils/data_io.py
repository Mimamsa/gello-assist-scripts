"""

"""
import os
import h5py
import pickle
import numpy as np
from datetime import datetime
from glob import glob
from natsort import natsorted


def load_pkl_demo(demo_dir):
    """Load demonstration data from pickle files.
    Args
        demo_dir (str): A directory contains pickle files of a demonstration.
    Returns
        (dict): A dictionary contains demonstration data.
            'timestamps': shape (N,) np.ndarray
            'wrist_rgb': shape (N, 720, 960, 3) np.ndarray
            'joint_positions': shape (N, 7) np.ndarray
            'joint_velocities': shape (N, 7) np.ndarray
            'control': shape (N, 7) np.ndarray
            'ee_pose': shape (N, 6) np.ndarray
    """
    pkl_paths = natsorted(glob(os.path.join(demo_dir, '*.pkl')))
    assert len(pkl_paths) > 30, 'Pickle files less than 30 files.'

    ret = {
        'timestamps': [],
        'wrist_rgb': [],
        #'base_rgb': [],
        'joint_positions': [],
        'joint_velocities': [],
        'control': [],
        'ee_pose': [],
    }

    for pkl_path in pkl_paths:
        # Extract timestamp
        # YYYY-MM-DDTHH:MM:SS.ffffff.pkl
        fname = os.path.basename(pkl_path)[:-4]
        dt = datetime.strptime(fname, '%Y-%m-%dT%H_%M_%S.%f')
        ts = dt.timestamp()
        ret['timestamps'].append(ts)

        # Extract frames
        with open(pkl_path, 'rb') as f:
            obs = pickle.load(f)

        for key in ['wrist_rgb','joint_positions','joint_velocities','control','ee_pose']:
            if type(obs[key]) is list:
                obj = np.array(obs[key])  # preventing obs[key] to be list
            else:
                obj = obs[key]
            ret[key].append(obj)

    for key in ret.keys():
        ret[key] = np.array(ret[key])

    return ret


def load_hdf5_demo(hdf5_path):
    """Load a demonstration data from a HDF5 File.
    Args
        hdf5_path (str): HDF5 demonstration file path.
    Returns
        (dict): Demonstration data.
            /action (N, 7)
            /observations/ee_pose (N, 6)
            /observations/images/wrist_rgb (N, 480, 640, 3)
            /observations/qpos (N, 7)
            /observations/qvel (N, 7)
            /timestamps (N,)
    """
    ret = {}
    keys = []
    with h5py.File(hdf5_path, 'r') as f:
        f.visit(lambda k: keys.append(k) if isinstance(f[k], h5py.Dataset) else None)
        for key in keys:
            ret[f[key].name] = f[key][()]
    return ret


def save_hdf5_demo(save_file, save_data, camera_names):
    """Save demonstration data as HDF5 File.
    Args
        save_file (str): Save file path.
        save_data (dict): Demonstration data to be saved.
            'timestamps': shape (N,) np.ndarray
            'wrist_rgb': shape (N, 720, 960, 3) np.ndarray
            'joint_positions': shape (N, 7) np.ndarray
            'joint_velocities': shape (N, 7) np.ndarray
            'control': shape (N, 7) np.ndarray
            'ee_pose': shape (N, 6) np.ndarray
        camera_names (list[str]): List of camera names.
    """
    max_timesteps = len(save_data['timestamps'])
    with h5py.File(save_file, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                     chunks=(1, 480, 640, 3), compression='gzip',compression_opts=9)
        ## compression='gzip',compression_opts=2,)
        ## compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        qpos = obs.create_dataset('qpos', (max_timesteps, 7))
        qvel = obs.create_dataset('qvel', (max_timesteps, 7))
        ee_pose = obs.create_dataset('ee_pose', (max_timesteps, 6))
        action = root.create_dataset('action', (max_timesteps, 7))
        timestamps = root.create_dataset('timestamps', (max_timesteps,))

        # save arrays
        for cam_name in camera_names:
            root['/observations/images/'+cam_name][...] = save_data[cam_name]
        root['/observations/qpos'][...] = save_data['joint_positions']
        root['/observations/qvel'][...] = save_data['joint_velocities']
        root['/observations/ee_pose'][...] = save_data['ee_pose']
        root['/action'][...] = save_data['control']
        root['/timestamps'][...] = save_data['timestamps']

        # for name, array in data_dict.items():
            # root[name][...] = array