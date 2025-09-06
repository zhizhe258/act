import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy, CupboardPolicy, StackPolicy

import IPython
e = IPython.embed


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    inject_noise = False
    render_cam_name = 'angle'
    
    # Set pose sampling mode
    pose_mode = args.get('pose_mode', 'random')
    import utils
    utils.POSE_SAMPLING_MODE = pose_mode
    print(f"Using pose mode: {pose_mode}")
    
    # Auto-generate dataset directory if not provided
    if dataset_dir is None:
        from constants import get_dataset_dir
        dataset_dir = get_dataset_dir(task_name, pose_mode)
        print(f"Auto-generated dataset directory: {dataset_dir}")

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'sim_cupboard_scripted':
        policy_cls = CupboardPolicy
    elif task_name == 'sim_stack_scripted':
        policy_cls = StackPolicy
    else:
        raise NotImplementedError

    success = []
    
    # Collect all episode positions for visualization
    all_peg_positions = []
    all_socket_positions = []
    all_cube_positions = []
    
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)
        
        # Collect current episode object positions
        if task_name == 'sim_insertion_scripted':
            # Get peg and socket positions from environment state
            env_state = ts.observation['env_state']
            if len(env_state) >= 14:  # Ensure we have peg and socket info (7+7=14 elements)
                # env_state[0:7] is peg pose, env_state[7:14] is socket pose
                peg_pos = env_state[0:3]  # First 3 elements are peg position
                socket_pos = env_state[7:10]  # Elements 8-10 are socket position
                all_peg_positions.append(peg_pos[:2])  # Only take XY coordinates
                all_socket_positions.append(socket_pos[:2])
                print(f"Episode {episode_idx} - Peg XY: {peg_pos[:2]}, Socket XY: {socket_pos[:2]}")
        elif task_name == 'sim_transfer_cube_scripted':
            # Get cube position from environment state
            env_state = ts.observation['env_state']
            if len(env_state) >= 7:  # Ensure we have cube info (7 elements)
                cube_pos = env_state[0:3]  # First 3 elements are cube position
                all_cube_positions.append(cube_pos[:2])  # Only take XY coordinates
                print(f"Episode {episode_idx} - Cube XY: {cube_pos[:2]}")
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0

        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')
    
    # Visualize position distribution for all episodes
    if len(all_peg_positions) > 0 or len(all_cube_positions) > 0:
        print("\nGenerating position distribution visualization...")
        
        if task_name == 'sim_insertion_scripted' and len(all_peg_positions) > 0:
            # Convert to numpy array
            peg_positions = np.array(all_peg_positions)
            socket_positions = np.array(all_socket_positions)
            
            # Create visualization plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot peg position distribution
            ax1.scatter(peg_positions[:, 0], peg_positions[:, 1], c='blue', alpha=0.7, s=50, label='Peg')
            ax1.set_xlabel('X coordinate')
            ax1.set_ylabel('Y coordinate')
            ax1.set_title(f'Peg Position Distribution (Total {len(peg_positions)} episodes)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add range boundaries
            import utils
            peg_x_range = utils.PEG_RANGES['x']
            peg_y_range = utils.PEG_RANGES['y']
            ax1.axvline(x=peg_x_range[0], color='red', linestyle='--', alpha=0.5, label=f'X range: {peg_x_range}')
            ax1.axvline(x=peg_x_range[1], color='red', linestyle='--', alpha=0.5)
            ax1.axhline(y=peg_y_range[0], color='red', linestyle='--', alpha=0.5, label=f'Y range: {peg_y_range}')
            ax1.axhline(y=peg_y_range[1], color='red', linestyle='--', alpha=0.5)
            
            # Plot socket position distribution
            ax2.scatter(socket_positions[:, 0], socket_positions[:, 1], c='green', alpha=0.7, s=50, label='Socket')
            ax2.set_xlabel('X coordinate')
            ax2.set_ylabel('Y coordinate')
            ax2.set_title(f'Socket Position Distribution (Total {len(socket_positions)} episodes)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add range boundaries
            socket_x_range = utils.SOCKET_RANGES['x']
            socket_y_range = utils.SOCKET_RANGES['y']
            ax2.axvline(x=socket_x_range[0], color='red', linestyle='--', alpha=0.5, label=f'X range: {socket_x_range}')
            ax2.axvline(x=socket_x_range[1], color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=socket_y_range[0], color='red', linestyle='--', alpha=0.5, label=f'Y range: {socket_y_range}')
            ax2.axhline(y=socket_y_range[1], color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save image
            plot_filename = os.path.join(dataset_dir, 'pose_distribution.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Position distribution plot saved to: {plot_filename}")
            
            # Display image
            plt.show()
            
            # Print statistics
            print(f"\nPosition Distribution Statistics:")
            print(f"Peg X range: [{peg_positions[:, 0].min():.4f}, {peg_positions[:, 0].max():.4f}]")
            print(f"Peg Y range: [{peg_positions[:, 1].min():.4f}, {peg_positions[:, 1].max():.4f}]")
            print(f"Socket X range: [{socket_positions[:, 0].min():.4f}, {socket_positions[:, 0].max():.4f}]")
            print(f"Socket Y range: [{socket_positions[:, 1].min():.4f}, {socket_positions[:, 1].max():.4f}]")
            
            # Check if within expected range
            peg_x_in_range = np.all((peg_positions[:, 0] >= peg_x_range[0]) & (peg_positions[:, 0] <= peg_x_range[1]))
            peg_y_in_range = np.all((peg_positions[:, 0] >= peg_y_range[0]) & (peg_positions[:, 0] <= peg_y_range[1]))
            socket_x_in_range = np.all((socket_positions[:, 0] >= socket_x_range[0]) & (socket_positions[:, 0] <= socket_x_range[1]))
            socket_y_in_range = np.all((socket_positions[:, 0] >= socket_y_range[0]) & (socket_positions[:, 0] <= socket_y_range[1]))
            
            print(f"\nRange Verification:")
            print(f"Peg X within range: {peg_x_in_range}")
            print(f"Peg Y within range: {peg_y_in_range}")
            print(f"Socket X within range: {socket_x_in_range}")
            print(f"Socket Y within range: {socket_y_in_range}")
            
        elif task_name == 'sim_transfer_cube_scripted' and len(all_cube_positions) > 0:
            # Convert to numpy array
            cube_positions = np.array(all_cube_positions)
            
            # Create visualization plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot cube position distribution
            ax.scatter(cube_positions[:, 0], cube_positions[:, 1], c='red', alpha=0.7, s=50, label='Cube')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_title(f'Cube Position Distribution (Total {len(cube_positions)} episodes)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add range boundaries
            import utils
            cube_x_range = utils.CUBE_RANGES['x']
            cube_y_range = utils.CUBE_RANGES['y']
            ax.axvline(x=cube_x_range[0], color='red', linestyle='--', alpha=0.5, label=f'X range: {cube_x_range}')
            ax.axvline(x=cube_x_range[1], color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=cube_y_range[0], color='red', linestyle='--', alpha=0.5, label=f'Y range: {cube_y_range}')
            ax.axhline(y=cube_y_range[1], color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save image
            plot_filename = os.path.join(dataset_dir, 'pose_distribution.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Position distribution plot saved to: {plot_filename}")
            
            # Display image
            plt.show()
            
            # Print statistics
            print(f"\nPosition Distribution Statistics:")
            print(f"Cube X range: [{cube_positions[:, 0].min():.4f}, {cube_positions[:, 0].max():.4f}]")
            print(f"Cube Y range: [{cube_positions[:, 1].min():.4f}, {cube_positions[:, 1].max():.4f}]")
            
            # Check if within expected range
            cube_x_in_range = np.all((cube_positions[:, 0] >= cube_x_range[0]) & (cube_positions[:, 0] <= cube_x_range[1]))
            cube_y_in_range = np.all((cube_positions[:, 1] >= cube_y_range[0]) & (cube_positions[:, 1] <= cube_y_range[1]))
            
            print(f"\nRange Verification:")
            print(f"Cube X within range: {cube_x_in_range}")
            print(f"Cube Y within range: {cube_y_in_range}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=False, default=None)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--pose_mode', action='store', type=str, choices=['fixed', 'random', 'edge', 'similar', 'uniform'], 
                        default='random', help='pose sampling mode: fixed, random, edge, or similar')
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))

