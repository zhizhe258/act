#!/usr/bin/env python3
"""
Analyze joint trajectory data for insertion task in three modes
Create trajectory comparison plots for 14 joints, showing mean and variance regions
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set font for plotting
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_episode_data(file_path):
    """Load hdf5 data for a single episode"""
    try:
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations/qpos'][:]  # shape: (episode_length, 14)
            return qpos
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def analyze_trajectory_statistics(all_qpos, mode_name):
    """Analyze statistical characteristics of trajectories"""
    if all_qpos is None or len(all_qpos) == 0:
        return None
    
    # Convert to numpy array shape: (num_episodes, episode_length, 14)
    all_qpos = np.array(all_qpos)
    
    # Calculate statistics for each time step
    episode_length = all_qpos.shape[1]
    num_joints = all_qpos.shape[2]
    
    trajectory_stats = {
        'mode': mode_name,
        'num_episodes': len(all_qpos),
        'episode_length': episode_length,
        'joints': {}
    }
    
    # Analyze each joint
    for joint_idx in range(num_joints):
        joint_data = all_qpos[:, :, joint_idx]  # shape: (num_episodes, episode_length)
        
        # Calculate statistics for each time step
        mean_trajectory = np.mean(joint_data, axis=0)  # shape: (episode_length,)
        std_trajectory = np.std(joint_data, axis=0)    # shape: (episode_length,)
        var_trajectory = np.var(joint_data, axis=0)    # shape: (episode_length,)
        
        # Calculate confidence interval (mean ± std)
        upper_bound = mean_trajectory + std_trajectory
        lower_bound = mean_trajectory - std_trajectory
        
        trajectory_stats['joints'][joint_idx] = {
            'mean': mean_trajectory,
            'std': std_trajectory,
            'var': var_trajectory,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'raw_data': joint_data
        }
    
    return trajectory_stats

def plot_joint_trajectories(all_stats):
    """Create trajectory comparison plots for 14 joints"""
    if not all_stats:
        print("No data to plot")
        return
    
    # Joint names
    joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
    
    # Create 14 subplots (7 left arm joints + 7 right arm joints)
    fig, axes = plt.subplots(7, 2, figsize=(20, 25))
    fig.suptitle('Joint Trajectory Analysis: Random vs Edge vs Similar Modes', fontsize=16)
    
    # Color settings
    colors = {'random': 'blue', 'edge': 'red', 'similar': 'green'}
    
    # Plot left arm joints (first column)
    for i, joint_name in enumerate(joint_names):
        ax = axes[i, 0]
        
        for stats in all_stats:
            if stats and i in stats['joints']:
                joint_data = stats['joints'][i]
                time_steps = np.arange(len(joint_data['mean']))
                
                # Plot mean trajectory
                ax.plot(time_steps, joint_data['mean'], 
                       color=colors[stats['mode']], 
                       label=f"{stats['mode']} (mean)", 
                       linewidth=2)
                
                # Plot variance region (mean ± std)
                ax.fill_between(time_steps, 
                              joint_data['lower_bound'], 
                              joint_data['upper_bound'], 
                              color=colors[stats['mode']], 
                              alpha=0.3, 
                              label=f"{stats['mode']} (±1σ)")
        
        ax.set_title(f'Left Arm: {joint_name}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot right arm joints (second column)
    for i, joint_name in enumerate(joint_names):
        ax = axes[i, 1]
        joint_idx = i + 7  # Right arm joint index starts from 7
        
        for stats in all_stats:
            if stats and joint_idx in stats['joints']:
                joint_data = stats['joints'][joint_idx]
                time_steps = np.arange(len(joint_data['mean']))
                
                # Plot mean trajectory
                ax.plot(time_steps, joint_data['mean'], 
                       color=colors[stats['mode']], 
                       label=f"{stats['mode']} (mean)", 
                       linewidth=2)
                
                # Plot variance region (mean ± std)
                ax.fill_between(time_steps, 
                              joint_data['lower_bound'], 
                              joint_data['upper_bound'], 
                              color=colors[stats['mode']], 
                              alpha=0.3, 
                              label=f"{stats['mode']} (±1σ)")
        
        ax.set_title(f'Right Arm: {joint_name}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('joint_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_joint_variance_comparison(all_stats):
    """Plot joint variance comparison chart"""
    if not all_stats:
        return
    
    joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
    
    # Create variance comparison chart
    fig, axes = plt.subplots(7, 2, figsize=(20, 25))
    
    
    colors = {'random': 'blue', 'edge': 'red', 'similar': 'green'}
    
    # Left arm joint variance
    for i, joint_name in enumerate(joint_names):
        ax = axes[i, 0]
        
        for stats in all_stats:
            if stats and i in stats['joints']:
                joint_data = stats['joints'][i]
                time_steps = np.arange(len(joint_data['var']))
                
                ax.plot(time_steps, joint_data['var'], 
                       color=colors[stats['mode']], 
                       label=f"{stats['mode']}", 
                       linewidth=2)
        
        ax.set_title(f'Left Arm: {joint_name} - Variance')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Right arm joint variance
    for i, joint_name in enumerate(joint_names):
        ax = axes[i, 1]
        joint_idx = i + 7
        
        for stats in all_stats:
            if stats and joint_idx in stats['joints']:
                joint_data = stats['joints'][joint_idx]
                time_steps = np.arange(len(joint_data['var']))
                
                ax.plot(time_steps, joint_data['var'], 
                       color=colors[stats['mode']], 
                       label=f"{stats['mode']}", 
                       linewidth=2)
        
        ax.set_title(f'Right Arm: {joint_name} - Variance')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('joint_variance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_statistics(all_stats):
    """Create summary statistics"""
    if not all_stats:
        return
    
    print("\n" + "="*100)
    print("TRAJECTORY ANALYSIS SUMMARY STATISTICS")
    print("="*100)
    
    joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
    
    for mode_stats in all_stats:
        if not mode_stats:
            continue
            
        print(f"\n{'='*80}")
        print(f"MODE: {mode_stats['mode'].upper()}")
        print(f"{'='*80}")
        print(f"Number of episodes: {mode_stats['num_episodes']}")
        print(f"Episode length: {mode_stats['episode_length']}")
        
        # Calculate overall statistics for each joint
        print(f"\n{'Joint':<15} {'Mean':<12} {'Std':<12} {'Var':<12} {'Range':<12}")
        print("-" * 70)
        
        for i, joint_name in enumerate(joint_names):
            if i in mode_stats['joints']:
                joint_data = mode_stats['joints'][i]
                left_joint = mode_stats['joints'][i]
                right_joint = mode_stats['joints'][i+7]
                
                # Left arm
                left_mean = np.mean(left_joint['mean'])
                left_std = np.mean(left_joint['std'])
                left_var = np.mean(left_joint['var'])
                left_range = np.max(left_joint['mean']) - np.min(left_joint['mean'])
                
                print(f"{joint_name:<15} {left_mean:<12.6f} {left_std:<12.6f} {left_var:<12.6f} {left_range:<12.6f}")
                
                # Right arm
                right_mean = np.mean(right_joint['mean'])
                right_std = np.mean(right_joint['std'])
                right_var = np.mean(right_joint['var'])
                right_range = np.max(right_joint['mean']) - np.min(right_joint['mean'])
                
                print(f"{'  (right)':<15} {right_mean:<12.6f} {right_std:<12.6f} {right_var:<12.6f} {right_range:<12.6f}")

def main():
    """Main function"""
    print("Starting analysis of joint trajectory data for insertion task in three modes...")
    
    # Data paths
    data_paths = {
        'random': 'data/sim_insertion_scripted',
        'edge': 'data/sim_insertion_scripted_edge', 
        'similar': 'data/sim_insertion_scripted_similar'
    }
    
    all_stats = []
    
    # Analyze each mode
    for mode_name, data_path in data_paths.items():
        print(f"\nAnalyzing {mode_name} mode...")
        
        # Get all episode files
        episode_files = sorted(list(Path(data_path).glob('episode_*.hdf5')))
        
        if not episode_files:
            print(f"No episode files found in {data_path}.")
            continue
            
        print(f"Found {len(episode_files)} episode files. Loading...")
        
        # Load all data
        all_qpos_list = []
        for i, file_path in enumerate(episode_files):
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i+1}/{len(episode_files)}...")
            qpos_data = load_episode_data(file_path)
            if qpos_data is not None:
                all_qpos_list.append(qpos_data)
        
        if not all_qpos_list:
            print(f"Failed to load any data from {mode_name} mode.")
            continue

        print(f"Successfully loaded data from {len(all_qpos_list)} episodes")
        
        # Analyze trajectory statistics
        stats = analyze_trajectory_statistics(all_qpos_list, mode_name)
        if stats:
            all_stats.append(stats)
            print(f"  {mode_name} mode analysis completed")
    
    # Create summary statistics
    create_summary_statistics(all_stats)
    
    # Plot trajectory comparison charts
    print("\nPlotting joint trajectory comparison charts...")
    plot_joint_trajectories(all_stats)
    
    # Plot variance comparison charts
    print("Plotting joint variance comparison charts...")
    plot_joint_variance_comparison(all_stats)
    
    print("\nAnalysis completed!")
    print("Generated files:")
    print("1. joint_trajectory_analysis.png - Joint trajectory comparison chart")
    print("2. joint_variance_comparison.png - Joint variance comparison chart")

if __name__ == "__main__":
    main() 
