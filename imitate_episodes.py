import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN, GYM_ALOHA_GRIPPER_TO_ACT_FN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
from auto_path_manager import create_organized_structure, print_structure_info

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def convert_16d_to_14d_qpos(qpos_16d):
    """Convert 16D qpos data to 14D for policy input (reverse of restore's convert_14d_to_16d_qpos)"""
    if len(qpos_16d) != 16:
        return qpos_16d
        
    # Create 14D state array
    qpos_14d = np.zeros(14)
    
    # Left arm joints (0-5)
    qpos_14d[0:6] = qpos_16d[0:6]
    
    # Left gripper: convert finger position back to normalized value [0,1]
    # Use the average of both fingers (they should be equal due to equality constraint)
    from constants import ACT_TO_GYM_ALOHA_GRIPPER_FN
    left_finger_pos = (qpos_16d[6] + qpos_16d[7]) / 2.0
    qpos_14d[6] = ACT_TO_GYM_ALOHA_GRIPPER_FN(left_finger_pos)
    qpos_14d[6] = np.clip(qpos_14d[6], 0.0, 1.0)
    
    # Right arm joints (7-12 -> 8-13)
    qpos_14d[7:13] = qpos_16d[8:14]
    
    # Right gripper: convert finger position back to normalized value [0,1]
    # Use the average of both fingers (they should be equal due to equality constraint)
    right_finger_pos = (qpos_16d[14] + qpos_16d[15]) / 2.0
    qpos_14d[13] = ACT_TO_GYM_ALOHA_GRIPPER_FN(right_finger_pos)
    qpos_14d[13] = np.clip(qpos_14d[13], 0.0, 1.0)
    
    return qpos_14d

def convert_14d_to_16d_action(action_14d):
    """Convert 14D action to 16D for environment execution (same as restore method)"""
    if len(action_14d) != 14:
        return action_14d
        
    # Create 16D joint position array
    action_16d = np.zeros(16)
    
    # Left arm joints (0-5)
    action_16d[0:6] = action_14d[0:6]
    
    # Left gripper: convert single normalized value to finger position
    left_gripper_norm = np.clip(action_14d[6], 0.0, 1.0)
    left_gripper_joint = GYM_ALOHA_GRIPPER_TO_ACT_FN(left_gripper_norm)
    action_16d[6] = left_gripper_joint     # left_left_finger
    action_16d[7] = left_gripper_joint     # left_right_finger
    
    # Right arm joints (8-13)
    action_16d[8:14] = action_14d[7:13]
    
    # Right gripper: convert single normalized value to finger position
    right_gripper_norm = np.clip(action_14d[13], 0.0, 1.0)
    right_gripper_joint = GYM_ALOHA_GRIPPER_TO_ACT_FN(right_gripper_norm)
    action_16d[14] = right_gripper_joint   # right_left_finger
    action_16d[15] = right_gripper_joint   # right_right_finger
    
    return action_16d

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # Auto-create organized directory structure
    if 'ckpt_dir' in args and args['ckpt_dir']:
        ckpt_dir = args['ckpt_dir']
        os.makedirs(ckpt_dir, exist_ok=True)
        paths = {'ckpt_dir': ckpt_dir}
    else:
        paths = create_organized_structure(task_name, policy_class)
        ckpt_dir = paths['ckpt_dir']
    
    # Print directory structure
    print_structure_info(paths)

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    print(f"DEBUG: task_name = {task_name}, is_sim = {is_sim}")
    if is_sim or task_name in ['bimanual_aloha_peg_insertion', 'bimanual_aloha_slot_insertion', 'converted_bimanual_aloha_slot_insertion', 'converted_bimanual_aloha_slot_insertion_with_vel_ACT', 'converted_bimanual_aloha_hook_package', 'converted_bimanual_aloha_thread_needle', 'converted_bimanual_aloha_peg_insertion', 'converted_bimanual_aloha_cube_transfer', 'trimanual_peg_insertion', 'trimanual_slot_insertion', 'trimanual_hook_package', 'trimanual_pour_test_tube', 'trimanual_thread_needle']:
        print(f"DEBUG: Using SIM_TASK_CONFIGS for {task_name}")
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
        # 如果任务在仿真任务列表中，确保is_sim为True
        is_sim = True
    else:
        print(f"DEBUG: Using TASK_CONFIGS for {task_name}")
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = args.get('backbone', 'resnet18')  # Allow backbone to be configurable
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }

    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'plot_kinematics': args.get('plot_kinematics', False)
    }

    if is_eval:
        # Auto-enable trajectory comparison for all eval runs
        config['plot_kinematics'] = True
        
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)

    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()

    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    plot_kinematics = config.get('plot_kinematics', False)
    onscreen_cam = 'angle'
    
    # Auto-create trajectory comparison directories for all evals
    from constants import JOINT_NAMES, DT
    
    # Create different folders based on temporal_agg setting
    if temporal_agg:
        comparison_base_dir = os.path.join(ckpt_dir, 'trajectory_comparisons_temporal_agg')
    else:
        comparison_base_dir = os.path.join(ckpt_dir, 'trajectory_comparisons_no_temporal_agg')
    
    os.makedirs(comparison_base_dir, exist_ok=True)
    
    # Create organized subdirectories
    success_dir = os.path.join(comparison_base_dir, 'successful_rollouts')
    failed_dir = os.path.join(comparison_base_dir, 'failed_rollouts')
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, weights_only=False))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        # 将数据集名称映射到仿真环境任务名称
        if 'converted_single_view_aloha_slot_insertion' in task_name:
            sim_task_name = 'bimanual_aloha_slot_insertion'
        elif 'converted_single_view_aloha_cube_transfer' in task_name:
            sim_task_name = 'bimanual_aloha_cube_transfer'
        elif 'converted_bimanual_aloha_hook_package' in task_name:
            sim_task_name = 'bimanual_aloha_hook_package'
        elif 'converted_bimanual_aloha_thread_needle' in task_name:
            sim_task_name = 'bimanual_aloha_thread_needle'
        elif 'converted_bimanual_aloha_peg_insertion' in task_name:
            sim_task_name = 'bimanual_aloha_peg_insertion'
        elif 'converted_bimanual_aloha_slot_insertion' in task_name:
            sim_task_name = 'bimanual_aloha_slot_insertion'
        elif 'converted_bimanual_aloha_cube_transfer' in task_name:
            sim_task_name = 'bimanual_aloha_pour_test_tube'  # 改为pour test tube
        else:
            sim_task_name = task_name
        
        env = make_sim_env(sim_task_name, enable_distractors=True)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()


        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        qvel_list = []  # Add velocity tracking
        target_qvel_list = []  # Add target velocity tracking  
        qaccel_list = []  # Add acceleration tracking
        target_qaccel_list = []  # Add target acceleration tracking
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qvel_numpy = np.array(obs['qvel'])  # Get real physics velocity from simulator
                
                # Convert 16D qpos to 14D for policy input if needed
                # (slot insertion uses BimanualAlohaTask which returns 14D directly)
                if len(qpos_numpy) == 16:
                    qpos_numpy_14d = convert_16d_to_14d_qpos(qpos_numpy)
                else:
                    qpos_numpy_14d = qpos_numpy
                
                qpos = pre_process(qpos_numpy_14d)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                # 将相机名称映射到仿真环境中的名称
                if 'converted_single_view_aloha_slot_insertion' in task_name or task_name.startswith('converted_bimanual_aloha_'):
                    sim_camera_names = ['overhead_cam']  # 仿真环境中的相机名称
                else:
                    sim_camera_names = camera_names
                
                curr_image = get_image(ts, sim_camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)

                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos_14d = action
                
                # Convert action format if needed based on environment type
                # (slot insertion uses BimanualAlohaTask which expects 14D actions directly)
                if len(qpos_numpy) == 16:
                    # For environments that return 16D observations (like restore_insertion)
                    target_qpos = convert_14d_to_16d_action(target_qpos_14d)
                else:
                    # For slot insertion and similar tasks that work with 14D
                    target_qpos = target_qpos_14d

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization - store real physics data and calculated target data
                # Store both in 14D format for consistent visualization
                if len(qpos_numpy) == 16:
                    qpos_list.append(qpos_numpy_14d)  # Use converted 14D version
                    target_qpos_list.append(target_qpos_14d)  # Use 14D version before conversion
                else:
                    qpos_list.append(qpos_numpy)
                    target_qpos_list.append(target_qpos)
                # Store velocity in 14D format for consistent visualization
                if len(qvel_numpy) == 16:
                    qvel_numpy_14d = convert_16d_to_14d_qpos(qvel_numpy)  # Reuse conversion for velocity
                    qvel_list.append(qvel_numpy_14d)
                else:
                    qvel_list.append(qvel_numpy)
                
                # Calculate target velocity using numerical differentiation
                if len(target_qpos_list) >= 2:
                    target_qvel = (target_qpos_list[-1] - target_qpos_list[-2]) / DT
                    target_qvel_list.append(target_qvel)
                else:
                    # First velocity step, set to zero
                    target_qvel_list.append(np.zeros_like(target_qpos_list[-1]))
                
                # Calculate accelerations using real physics velocities vs target velocities
                if len(qvel_list) >= 2:
                    # Real acceleration from real physics velocities
                    current_qaccel = (qvel_list[-1] - qvel_list[-2]) / DT
                    qaccel_list.append(current_qaccel)
                    
                    # Target acceleration from target velocities
                    if len(target_qvel_list) >= 2:
                        target_qaccel = (target_qvel_list[-1] - target_qvel_list[-2]) / DT
                        target_qaccel_list.append(target_qaccel)
                    else:
                        target_qaccel_list.append(np.zeros_like(target_qvel_list[-1]))
                else:
                    # First acceleration step, set to zero
                    qaccel_list.append(np.zeros_like(qvel_list[-1]))
                    target_qaccel_list.append(np.zeros_like(target_qvel_list[-1]))
                
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        # Auto-save trajectory comparison for all rollouts
        success = episode_highest_reward == env_max_reward
        category_dir = success_dir if success else failed_dir
        rollout_dir = os.path.join(category_dir, f'rollout_{rollout_id:03d}_return_{episode_return:.2f}')
        plot_trajectory_comparison(qpos_list, target_qpos_list, qvel_list, target_qvel_list,
                                 qaccel_list, target_qaccel_list, rollout_dir, 
                                 episode_return, success, DT)
        
        # Save videos for each rollout
        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(rollout_dir, f'video.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # Generate overall trajectory analysis summary
    generate_trajectory_summary(comparison_base_dir, success_dir, failed_dir, 
                               episode_returns, highest_rewards, success_rate, avg_return)

    # save success rate to txt with temporal_agg distinction
    temporal_suffix = '_temporal_agg' if temporal_agg else '_no_temporal_agg'
    result_file_name = 'result_' + ckpt_name.split('.')[0] + temporal_suffix + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def generate_trajectory_summary(base_dir, success_dir, failed_dir, episode_returns, 
                               highest_rewards, success_rate, avg_return):
    """
    Generate overall trajectory analysis summary and comparison between successful and failed runs
    """
    import matplotlib.pyplot as plt
    import json
    import glob
    
    # Collect all trajectory metrics
    success_metrics = []
    failed_metrics = []
    
    # Parse successful rollouts
    for metrics_file in glob.glob(os.path.join(success_dir, '**/trajectory_metrics.json'), recursive=True):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            success_metrics.append(metrics)
    
    # Parse failed rollouts
    for metrics_file in glob.glob(os.path.join(failed_dir, '**/trajectory_metrics.json'), recursive=True):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            failed_metrics.append(metrics)
    
    # Generate comparison plots
    if success_metrics and failed_metrics:
        plot_success_vs_failure_analysis(success_metrics, failed_metrics, base_dir)
    
    # Generate overall summary report
    summary_report = {
        'evaluation_summary': {
            'total_rollouts': len(episode_returns),
            'success_rate': float(success_rate),
            'average_return': float(avg_return),
            'successful_rollouts': len(success_metrics),
            'failed_rollouts': len(failed_metrics)
        },
        'trajectory_analysis': {
            'successful_runs': {
                'avg_pos_rms': np.mean([m['overall_pos_rms'] for m in success_metrics]) if success_metrics else 0,
                'avg_vel_rms': np.mean([m['overall_vel_rms'] for m in success_metrics]) if success_metrics else 0,
                'avg_return': np.mean([m['episode_return'] for m in success_metrics]) if success_metrics else 0
            },
            'failed_runs': {
                'avg_pos_rms': np.mean([m['overall_pos_rms'] for m in failed_metrics]) if failed_metrics else 0,
                'avg_vel_rms': np.mean([m['overall_vel_rms'] for m in failed_metrics]) if failed_metrics else 0,
                'avg_return': np.mean([m['episode_return'] for m in failed_metrics]) if failed_metrics else 0
            }
        }
    }
    
    # Save summary report
    summary_path = os.path.join(base_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"\n📊 Trajectory analysis summary saved to: {summary_path}")
    print(f"📁 Successful rollouts: {len(success_metrics)} (in {success_dir})")
    print(f"📁 Failed rollouts: {len(failed_metrics)} (in {failed_dir})")


def plot_success_vs_failure_analysis(success_metrics, failed_metrics, save_dir):
    """
    Generate comparison plots between successful and failed rollouts
    """
    import matplotlib.pyplot as plt
    
    # Extract metrics for comparison
    success_pos_rms = [m['overall_pos_rms'] for m in success_metrics]
    success_vel_rms = [m['overall_vel_rms'] for m in success_metrics] 
    success_returns = [m['episode_return'] for m in success_metrics]
    
    failed_pos_rms = [m['overall_pos_rms'] for m in failed_metrics]
    failed_vel_rms = [m['overall_vel_rms'] for m in failed_metrics]
    failed_returns = [m['episode_return'] for m in failed_metrics]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Trajectory Analysis: Successful vs Failed Rollouts', fontsize=16, fontweight='bold')
    
    # Position RMS comparison
    axes[0,0].hist(success_pos_rms, alpha=0.7, label='Successful', bins=10, color='green')
    axes[0,0].hist(failed_pos_rms, alpha=0.7, label='Failed', bins=10, color='red')
    axes[0,0].set_xlabel('Position RMS Error')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('Position RMS Error Distribution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Velocity RMS comparison
    axes[0,1].hist(success_vel_rms, alpha=0.7, label='Successful', bins=10, color='green')
    axes[0,1].hist(failed_vel_rms, alpha=0.7, label='Failed', bins=10, color='red')
    axes[0,1].set_xlabel('Velocity RMS Error')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Velocity RMS Error Distribution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Return comparison
    axes[1,0].hist(success_returns, alpha=0.7, label='Successful', bins=10, color='green')
    axes[1,0].hist(failed_returns, alpha=0.7, label='Failed', bins=10, color='red')
    axes[1,0].set_xlabel('Episode Return')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Episode Return Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Scatter plot: Position RMS vs Return
    axes[1,1].scatter(success_pos_rms, success_returns, alpha=0.7, label='Successful', color='green', s=50)
    axes[1,1].scatter(failed_pos_rms, failed_returns, alpha=0.7, label='Failed', color='red', s=50)
    axes[1,1].set_xlabel('Position RMS Error')
    axes[1,1].set_ylabel('Episode Return')
    axes[1,1].set_title('Position Error vs Episode Return')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(save_dir, 'success_vs_failure_analysis.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Success vs failure analysis saved to: {comparison_path}")


def plot_trajectory_comparison(qpos_executed, qpos_predicted, qvel_executed, qvel_predicted, 
                               qaccel_executed, qaccel_predicted, save_dir, episode_return, success, dt):
    """
    Plot comparison between executed trajectory and predicted trajectory
    Auto-generates organized folder structure for analysis
    """
    from constants import JOINT_NAMES
    
    os.makedirs(save_dir, exist_ok=True)
    left_arm_dir = os.path.join(save_dir, 'left_arm')
    right_arm_dir = os.path.join(save_dir, 'right_arm')
    error_dir = os.path.join(save_dir, 'error_analysis')
    
    os.makedirs(left_arm_dir, exist_ok=True)
    os.makedirs(right_arm_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    
    qpos_executed = np.array(qpos_executed)
    qpos_predicted = np.array(qpos_predicted)
    qvel_executed = np.array(qvel_executed)
    qvel_predicted = np.array(qvel_predicted) 
    qaccel_executed = np.array(qaccel_executed)
    qaccel_predicted = np.array(qaccel_predicted)
    
    # Split data for left and right arms (7 states per arm: 6 joints + gripper)
    STATE_NAMES = JOINT_NAMES + ["gripper"]
    
    left_qpos_exec = qpos_executed[:, :7]
    left_qpos_pred = qpos_predicted[:, :7]
    left_qvel_exec = qvel_executed[:, :7]
    left_qvel_pred = qvel_predicted[:, :7]
    left_qaccel_exec = qaccel_executed[:, :7]
    left_qaccel_pred = qaccel_predicted[:, :7]
    
    right_qpos_exec = qpos_executed[:, 7:14]
    right_qpos_pred = qpos_predicted[:, 7:14]
    right_qvel_exec = qvel_executed[:, 7:14]
    right_qvel_pred = qvel_predicted[:, 7:14]
    right_qaccel_exec = qaccel_executed[:, 7:14]
    right_qaccel_pred = qaccel_predicted[:, 7:14]
    
    time_steps = np.arange(len(qpos_executed)) * dt
    
    # Plot for left arm
    plot_arm_trajectory_comparison(left_qpos_exec, left_qpos_pred, 
                                 left_qvel_exec, left_qvel_pred,
                                 left_qaccel_exec, left_qaccel_pred,
                                 time_steps, 'Left Arm', left_arm_dir, 
                                 STATE_NAMES, episode_return, success)
    
    # Plot for right arm
    plot_arm_trajectory_comparison(right_qpos_exec, right_qpos_pred,
                                 right_qvel_exec, right_qvel_pred, 
                                 right_qaccel_exec, right_qaccel_pred,
                                 time_steps, 'Right Arm', right_arm_dir,
                                 STATE_NAMES, episode_return, success)
    
    # Generate error analysis
    plot_error_analysis(qpos_executed, qpos_predicted, qvel_executed, qvel_predicted,
                       qaccel_executed, qaccel_predicted, time_steps, error_dir,
                       episode_return, success)


def plot_arm_trajectory_comparison(qpos_exec, qpos_pred, qvel_exec, qvel_pred, 
                                 qaccel_exec, qaccel_pred, time_steps, arm_name, 
                                 save_dir, joint_names, episode_return, success):
    """
    Plot trajectory comparison for one arm
    """
    import matplotlib.pyplot as plt
    
    num_joints = qpos_exec.shape[1]
    success_str = "SUCCESS" if success else "FAILED"
    
    # Individual joint comparison plots
    for joint_idx in range(num_joints):
        joint_name = joint_names[joint_idx]
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'{arm_name} - {joint_name} | Return: {episode_return:.2f} | {success_str}', 
                    fontsize=16, fontweight='bold')
        
        # Position comparison
        axes[0].plot(time_steps, qpos_exec[:, joint_idx], 'b-', linewidth=2, label='Executed', alpha=0.8)
        axes[0].plot(time_steps, qpos_pred[:, joint_idx], 'r--', linewidth=2, label='Predicted', alpha=0.8)
        axes[0].fill_between(time_steps, qpos_exec[:, joint_idx], qpos_pred[:, joint_idx], 
                           alpha=0.3, color='orange', label='Error')
        axes[0].set_ylabel('Position (rad)', fontsize=12)
        axes[0].set_title('Joint Position Comparison')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Velocity comparison
        axes[1].plot(time_steps, qvel_exec[:, joint_idx], 'b-', linewidth=2, label='Executed', alpha=0.8)
        axes[1].plot(time_steps, qvel_pred[:, joint_idx], 'r--', linewidth=2, label='Predicted', alpha=0.8)
        axes[1].fill_between(time_steps, qvel_exec[:, joint_idx], qvel_pred[:, joint_idx],
                           alpha=0.3, color='orange', label='Error')
        axes[1].set_ylabel('Velocity (rad/s)', fontsize=12)
        axes[1].set_title('Joint Velocity Comparison')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Acceleration comparison
        axes[2].plot(time_steps, qaccel_exec[:, joint_idx], 'b-', linewidth=2, label='Executed', alpha=0.8)
        axes[2].plot(time_steps, qaccel_pred[:, joint_idx], 'r--', linewidth=2, label='Predicted', alpha=0.8)
        axes[2].fill_between(time_steps, qaccel_exec[:, joint_idx], qaccel_pred[:, joint_idx],
                           alpha=0.3, color='orange', label='Error')
        axes[2].set_ylabel('Acceleration (rad/s²)', fontsize=12)
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].set_title('Joint Acceleration Comparison')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'{joint_name}_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Combined overview plot
    fig, axes = plt.subplots(3, 1, figsize=(18, 15))
    fig.suptitle(f'{arm_name} - All Joints Overview | Return: {episode_return:.2f} | {success_str}', 
                fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_joints))
    
    for joint_idx in range(num_joints):
        joint_name = joint_names[joint_idx]
        color = colors[joint_idx]
        
        # Position
        axes[0].plot(time_steps, qpos_exec[:, joint_idx], color=color, linewidth=2, 
                    alpha=0.8, label=f'{joint_name} (exec)')
        axes[0].plot(time_steps, qpos_pred[:, joint_idx], color=color, linewidth=2, 
                    linestyle='--', alpha=0.6, label=f'{joint_name} (pred)')
        
        # Velocity
        axes[1].plot(time_steps, qvel_exec[:, joint_idx], color=color, linewidth=2,
                    alpha=0.8, label=f'{joint_name} (exec)')
        axes[1].plot(time_steps, qvel_pred[:, joint_idx], color=color, linewidth=2,
                    linestyle='--', alpha=0.6, label=f'{joint_name} (pred)')
        
        # Acceleration
        axes[2].plot(time_steps, qaccel_exec[:, joint_idx], color=color, linewidth=2,
                    alpha=0.8, label=f'{joint_name} (exec)')
        axes[2].plot(time_steps, qaccel_pred[:, joint_idx], color=color, linewidth=2,
                    linestyle='--', alpha=0.6, label=f'{joint_name} (pred)')
    
    axes[0].set_ylabel('Position (rad)', fontsize=12)
    axes[0].set_title('Joint Positions: Solid=Executed, Dashed=Predicted')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    axes[1].set_ylabel('Velocity (rad/s)', fontsize=12)
    axes[1].set_title('Joint Velocities: Solid=Executed, Dashed=Predicted')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    axes[2].set_ylabel('Acceleration (rad/s²)', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_title('Joint Accelerations: Solid=Executed, Dashed=Predicted')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    overview_path = os.path.join(save_dir, 'all_joints_comparison_overview.png')
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_analysis(qpos_exec, qpos_pred, qvel_exec, qvel_pred, 
                       qaccel_exec, qaccel_pred, time_steps, save_dir,
                       episode_return, success):
    """
    Generate comprehensive error analysis plots and metrics
    """
    import matplotlib.pyplot as plt
    import json
    from constants import JOINT_NAMES
    
    success_str = "SUCCESS" if success else "FAILED"
    STATE_NAMES = JOINT_NAMES + ["gripper"]
    
    # Calculate errors
    pos_error = qpos_exec - qpos_pred
    vel_error = qvel_exec - qvel_pred
    accel_error = qaccel_exec - qaccel_pred
    
    # Error metrics
    pos_mse = np.mean(pos_error**2, axis=0)
    vel_mse = np.mean(vel_error**2, axis=0)
    accel_mse = np.mean(accel_error**2, axis=0)
    
    pos_mae = np.mean(np.abs(pos_error), axis=0)
    vel_mae = np.mean(np.abs(vel_error), axis=0)
    accel_mae = np.mean(np.abs(accel_error), axis=0)
    
    # RMS error over time
    pos_rms_time = np.sqrt(np.mean(pos_error**2, axis=1))
    vel_rms_time = np.sqrt(np.mean(vel_error**2, axis=1))
    accel_rms_time = np.sqrt(np.mean(accel_error**2, axis=1))
    
    # Plot error over time
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'RMS Error Analysis | Return: {episode_return:.2f} | {success_str}', 
                fontsize=16, fontweight='bold')
    
    axes[0].plot(time_steps, pos_rms_time, 'r-', linewidth=2)
    axes[0].set_ylabel('Position RMS Error (rad)', fontsize=12)
    axes[0].set_title('Position RMS Error Over Time')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_steps, vel_rms_time, 'g-', linewidth=2)
    axes[1].set_ylabel('Velocity RMS Error (rad/s)', fontsize=12)
    axes[1].set_title('Velocity RMS Error Over Time')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time_steps, accel_rms_time, 'b-', linewidth=2)
    axes[2].set_ylabel('Acceleration RMS Error (rad/s²)', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_title('Acceleration RMS Error Over Time')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    error_time_path = os.path.join(save_dir, 'rms_error_over_time.png')
    plt.savefig(error_time_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot joint-wise error distribution
    all_joint_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    fig.suptitle(f'Joint-wise Error Metrics | Return: {episode_return:.2f} | {success_str}', 
                fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(all_joint_names))
    
    axes[0].bar(x_pos, pos_mse, alpha=0.7, color='red')
    axes[0].set_ylabel('Position MSE (rad²)', fontsize=12)
    axes[0].set_title('Position Mean Squared Error by Joint')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(all_joint_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(x_pos, vel_mse, alpha=0.7, color='green')
    axes[1].set_ylabel('Velocity MSE (rad²/s²)', fontsize=12)
    axes[1].set_title('Velocity Mean Squared Error by Joint')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(all_joint_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].bar(x_pos, accel_mse, alpha=0.7, color='blue')
    axes[2].set_ylabel('Acceleration MSE (rad²/s⁴)', fontsize=12)
    axes[2].set_title('Acceleration Mean Squared Error by Joint')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(all_joint_names, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    error_joints_path = os.path.join(save_dir, 'joint_wise_error_metrics.png')
    plt.savefig(error_joints_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save metrics to JSON
    metrics = {
        'episode_return': float(episode_return),
        'success': bool(success),
        'position_mse': pos_mse.tolist(),
        'velocity_mse': vel_mse.tolist(), 
        'acceleration_mse': accel_mse.tolist(),
        'position_mae': pos_mae.tolist(),
        'velocity_mae': vel_mae.tolist(),
        'acceleration_mae': accel_mae.tolist(),
        'overall_pos_rms': float(np.mean(pos_rms_time)),
        'overall_vel_rms': float(np.mean(vel_rms_time)),
        'overall_accel_rms': float(np.mean(accel_rms_time)),
        'joint_names': all_joint_names
    }
    
    metrics_path = os.path.join(save_dir, 'trajectory_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f'Trajectory comparison saved to: {save_dir}')


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    

    
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        

        
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir (optional, auto-generated if not provided)', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--plot_kinematics', action='store_true', help='Plot trajectory comparison during eval')
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=100, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    

    
    # backbone selection
    parser.add_argument('--backbone', action='store', type=str, help='backbone model (resnet18/34/50/101 or dinov2_vits14/vitb14/vitl14/vitg14)', 
                        default='resnet18', required=False)
    
    main(vars(parser.parse_args()))
