#!/usr/bin/env python3
"""
Bimanual ALOHA Evaluation Script
===============================

Separate evaluation script for trained bimanual ALOHA policies.
This script handles MuJoCo environment interaction for evaluation only.
"""

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange

# Environment and evaluation imports
from sim_env import make_sim_env
from constants import DT, JOINT_NAMES
from utils import set_seed
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

import IPython
e = IPython.embed

# Bimanual task configurations (same as training script)
BIMANUAL_TASK_CONFIGS = {
    'bimanual_aloha_cube_transfer': {
        'env_name': 'bimanual_aloha_cube_transfer',
        'state_dim': 14,
        'action_dim': 14,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
    },
    'bimanual_aloha_peg_insertion': {
        'env_name': 'bimanual_aloha_peg_insertion',
        'state_dim': 14,
        'action_dim': 14,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
    },
    'bimanual_aloha_slot_insertion': {
        'env_name': 'bimanual_aloha_slot_insertion',
        'state_dim': 14,
        'action_dim': 14,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
    },
    'converted_bimanual_aloha_slot_insertion_with_vel': {
        'env_name': 'bimanual_aloha_slot_insertion',  # Use same environment
        'state_dim': 14,
        'action_dim': 14,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
    },
    'bimanual_aloha_color_cubes': {
        'env_name': 'bimanual_aloha_color_cubes',
        'state_dim': 14,
        'action_dim': 14,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
    },
    'bimanual_aloha_hook_package': {
        'env_name': 'bimanual_aloha_hook_package',
        'state_dim': 14,
        'action_dim': 14,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
    },
    'bimanual_aloha_pour_test_tube': {
        'env_name': 'bimanual_aloha_pour_test_tube',
        'state_dim': 14,
        'action_dim': 14,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
    },
    'bimanual_aloha_thread_needle': {
        'env_name': 'bimanual_aloha_thread_needle',
        'state_dim': 14,
        'action_dim': 14,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
    },
}

def make_policy(policy_class, policy_config):
    """Create policy instance"""
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'RTC_Improved':
        from policy_rtc_improved import RTCPolicy_Improved
        policy = RTCPolicy_Improved(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError(f"Policy class {policy_class} not supported")
    return policy

def get_image(ts, camera_names):
    """Extract and preprocess images from timestep"""
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def eval_bc(args):
    """Evaluate trained behavior cloning policy"""
    set_seed(1000)
    
    # Parse arguments
    ckpt_dir = args.ckpt_dir
    task_name = args.task_name
    policy_class = args.policy_class
    temporal_agg = args.temporal_agg
    num_rollouts = args.num_rollouts
    save_video = args.save_video
    onscreen_render = args.onscreen_render
    
    # Get task configuration
    if task_name not in BIMANUAL_TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}")
    
    task_config = BIMANUAL_TASK_CONFIGS[task_name]
    env_name = task_config['env_name']
    state_dim = task_config['state_dim']
    camera_names = task_config['camera_names']
    
    print(f"🎯 Evaluating: {task_name}")
    print(f"🧠 Policy: {policy_class}")
    print(f"📁 Checkpoint dir: {ckpt_dir}")
    print(f"🔄 Temporal aggregation: {temporal_agg}")
    print(f"🎬 Number of rollouts: {num_rollouts}")
    
    # Load policy configuration and stats
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Dataset stats not found: {stats_path}")
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    # Create minimal policy config for evaluation
    policy_config = {
        'camera_names': camera_names,
        'state_dim': state_dim,
    }
    
    # Load policy checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Policy checkpoint not found: {ckpt_path}")
    
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, weights_only=False))
    print(f"📦 Policy loaded: {loading_status}")
    policy.cuda()
    policy.eval()
    
    # Data preprocessing functions
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    # Create environment
    print(f"🌍 Creating environment: {env_name}")
    env = make_sim_env(env_name)
    env_max_reward = env.task.max_reward
    
    # Configure evaluation parameters
    query_frequency = policy_config.get('num_queries', 100)
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config.get('num_queries', 100)
    
    max_timesteps = 500  # Default episode length
    
    # Evaluation loop
    episode_returns = []
    highest_rewards = []
    
    print(f"\n🚀 Starting evaluation...")
    
    for rollout_id in tqdm(range(num_rollouts), desc="Evaluating"):
        # Reset environment
        ts = env.reset()
        
        # Initialize temporal aggregation if needed
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()
        
        # Episode data storage
        image_list = []
        qpos_list = []
        rewards = []
        
        # Onscreen rendering setup
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id='overhead_cam'))
            plt.ion()
        
        # Episode loop
        with torch.inference_mode():
            for t in range(max_timesteps):
                # Update onscreen render
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id='overhead_cam')
                    plt_img.set_data(image)
                    plt.pause(DT)
                
                # Process observation
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                
                qpos_numpy = np.array(obs['qpos'])
                qpos_list.append(qpos_numpy)
                
                # Preprocess state
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                curr_image = get_image(ts, camera_names)
                
                # Query policy
                if policy_class == "ACT":
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
                elif policy_class == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                elif policy_class == "RTC_Improved":
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
                else:
                    raise NotImplementedError
                
                # Post-process action
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                
                # Step environment
                ts = env.step(action)
                rewards.append(ts.reward)
        
        if onscreen_render:
            plt.close()
        
        # Calculate episode metrics
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        
        success = episode_highest_reward == env_max_reward
        print(f'Rollout {rollout_id}: Return={episode_return:.2f}, Max Reward={episode_highest_reward:.2f}, Success={success}')
        
        # Save video if requested
        if save_video:
            video_dir = os.path.join(ckpt_dir, 'evaluation_videos')
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f'rollout_{rollout_id:03d}_return_{episode_return:.2f}.mp4')
            save_videos(image_list, DT, video_path=video_path)
    
    # Calculate final statistics
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Success rate: {success_rate:.3f}")
    print(f"   Average return: {avg_return:.3f}")
    print(f"   Max possible reward: {env_max_reward}")
    
    # Save results
    results = {
        'success_rate': success_rate,
        'avg_return': avg_return,
        'episode_returns': episode_returns,
        'highest_rewards': highest_rewards,
        'env_max_reward': env_max_reward,
        'task_name': task_name,
        'policy_class': policy_class,
        'temporal_agg': temporal_agg,
        'num_rollouts': num_rollouts,
    }
    
    temporal_suffix = '_temporal_agg' if temporal_agg else '_no_temporal_agg'
    results_path = os.path.join(ckpt_dir, f'evaluation_results{temporal_suffix}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"💾 Results saved to: {results_path}")
    
    return success_rate, avg_return

def main():
    parser = argparse.ArgumentParser(description='Bimanual ALOHA Evaluation Script')
    
    # Required arguments
    parser.add_argument('--ckpt_dir', type=str, required=True,
                       help='Directory containing trained policy checkpoint')
    parser.add_argument('--task_name', type=str, required=True,
                       choices=list(BIMANUAL_TASK_CONFIGS.keys()),
                       help='Task name for evaluation')
    parser.add_argument('--policy_class', type=str, required=True,
                       choices=['ACT', 'RTC_Improved', 'CNNMLP'],
                       help='Policy class')
    
    # Optional arguments
    parser.add_argument('--temporal_agg', action='store_true',
                       help='Enable temporal aggregation during evaluation')
    parser.add_argument('--num_rollouts', type=int, default=50,
                       help='Number of evaluation rollouts')
    parser.add_argument('--save_video', action='store_true',
                       help='Save videos of evaluation rollouts')
    parser.add_argument('--onscreen_render', action='store_true',
                       help='Render evaluation on screen')
    
    args = parser.parse_args()
    
    # Run evaluation
    try:
        success_rate, avg_return = eval_bc(args)
        print(f"\n✅ Evaluation completed successfully!")
        print(f"   Final success rate: {success_rate:.3f}")
        print(f"   Final average return: {avg_return:.3f}")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        raise

if __name__ == '__main__':
    main()