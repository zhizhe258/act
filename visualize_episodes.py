import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from constants import DT

import IPython
e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, action, image_dict

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'

    qpos, qvel, action, image_dict = load_hdf5(dataset_dir, dataset_name)
    save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
    
    # Plot joint kinematics for both arms
    plot_joint_kinematics(qpos, qvel, dataset_dir, dataset_name)
    # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back


def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()

def plot_joint_kinematics(qpos, qvel, dataset_dir, dataset_name):
    """
    Plot joint position, velocity, and acceleration for both arms
    Creates organized folder structure: episode_X/left_arm/ and episode_X/right_arm/
    """
    # Create output directory structure
    episode_dir = os.path.join(dataset_dir, f'{dataset_name}_joint_plots')
    left_arm_dir = os.path.join(episode_dir, 'left_arm')
    right_arm_dir = os.path.join(episode_dir, 'right_arm')
    
    os.makedirs(left_arm_dir, exist_ok=True)
    os.makedirs(right_arm_dir, exist_ok=True)
    
    qpos = np.array(qpos)
    qvel = np.array(qvel)
    
    # Calculate acceleration using finite differences
    dt = DT
    qaccel = np.zeros_like(qvel)
    qaccel[1:] = (qvel[1:] - qvel[:-1]) / dt
    
    # Split data for left and right arms
    # Based on the code structure: 7 states per arm (6 joints + gripper)
    # Order: left_arm (0:7), right_arm (7:14)
    left_qpos = qpos[:, :7]
    left_qvel = qvel[:, :7] 
    left_qaccel = qaccel[:, :7]
    
    right_qpos = qpos[:, 7:14]
    right_qvel = qvel[:, 7:14]
    right_qaccel = qaccel[:, 7:14]
    
    # Time vector
    time_steps = np.arange(len(qpos)) * dt
    
    # Plot for left arm
    plot_arm_kinematics(left_qpos, left_qvel, left_qaccel, time_steps, 
                       'Left Arm', left_arm_dir, STATE_NAMES)
    
    # Plot for right arm  
    plot_arm_kinematics(right_qpos, right_qvel, right_qaccel, time_steps,
                       'Right Arm', right_arm_dir, STATE_NAMES)
    
    print(f'Joint kinematic plots saved to: {episode_dir}')

def plot_arm_kinematics(qpos, qvel, qaccel, time_steps, arm_name, save_dir, joint_names):
    """
    Plot position, velocity, and acceleration for one arm
    """
    num_joints = qpos.shape[1]
    
    # Create individual plots for each joint
    for joint_idx in range(num_joints):
        joint_name = joint_names[joint_idx]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'{arm_name} - {joint_name}', fontsize=16, fontweight='bold')
        
        # Position
        axes[0].plot(time_steps, qpos[:, joint_idx], 'b-', linewidth=2)
        axes[0].set_ylabel('Position (rad)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Joint Position')
        
        # Velocity
        axes[1].plot(time_steps, qvel[:, joint_idx], 'g-', linewidth=2)
        axes[1].set_ylabel('Velocity (rad/s)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Joint Velocity')
        
        # Acceleration
        axes[2].plot(time_steps, qaccel[:, joint_idx], 'r-', linewidth=2)
        axes[2].set_ylabel('Acceleration (rad/s²)', fontsize=12)
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Joint Acceleration')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, f'{joint_name}_kinematics.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create combined overview plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'{arm_name} - All Joints Overview', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_joints))
    
    for joint_idx in range(num_joints):
        joint_name = joint_names[joint_idx]
        color = colors[joint_idx]
        
        # Position
        axes[0].plot(time_steps, qpos[:, joint_idx], color=color, 
                    linewidth=2, label=joint_name)
        
        # Velocity  
        axes[1].plot(time_steps, qvel[:, joint_idx], color=color,
                    linewidth=2, label=joint_name)
        
        # Acceleration
        axes[2].plot(time_steps, qaccel[:, joint_idx], color=color,
                    linewidth=2, label=joint_name)
    
    axes[0].set_ylabel('Position (rad)', fontsize=12)
    axes[0].set_title('Joint Positions')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    axes[1].set_ylabel('Velocity (rad/s)', fontsize=12) 
    axes[1].set_title('Joint Velocities')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    axes[2].set_ylabel('Acceleration (rad/s²)', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_title('Joint Accelerations')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save combined plot
    overview_path = os.path.join(save_dir, 'all_joints_overview.png')
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))
