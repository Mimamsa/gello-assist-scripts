""" Image & motion trajectory Visualize module

"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from scipy.spatial.transform import Rotation as R


def calculate_endpoint(tcp_poses):
    rotation_matrix = R.from_rotvec(tcp_poses[:,3:]).as_matrix()
    unit_vector = np.array([0, 0, 0.1])
    endpoints = rotation_matrix @ unit_vector
    start = tcp_poses[:,:3]
    return start + endpoints

def update(num, data, line, quiver):
    line.set_data(data[:num, 0], data[:num, 1])
    line.set_3d_properties(data[:num, 2])

    # quiver.remove()
    # convert to segment
    seg = data[num,:]
    quiver.set_segments([[seg[:3], seg[3:]]])

def display_tcp_pos(tcp_poses, fps=60):
    """
    ref:
    https://stackoverflow.com/questions/48911643/set-uvc-equivilent-for-a-3d-quiver-plot-in-matplotlib
    """
    interval = 1/fps
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    endpoints = calculate_endpoint(tcp_poses)
    tcp_poses = np.concatenate([tcp_poses[:,:3], endpoints], axis=-1)

    start_pose = tcp_poses[0,:]
    num_steps = len(tcp_poses)
    line, = ax.plot(start_pose[0], start_pose[1], start_pose[2])
    quiver = ax.quiver(*start_pose, colors='r')

    ## Setting the axes properties
    ax.set_xlim3d([-1.0, 0.5])
    ax.set_xlabel('X')

    ax.set_ylim3d([-0.3, 0.3])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-0.3, 0.3])
    ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, update, num_steps, fargs=(tcp_poses, line, quiver), interval=interval, blit=False)
    ##ani.save('matplot003.gif', writer='imagemagick')
    plt.show()