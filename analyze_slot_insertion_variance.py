#!/usr/bin/env python3
"""
Analyze joint variance in the converted_bimanual_aloha_slot_insertion dataset
Based on the implementation approach from /home/zzt/act11/trajectory_analysis_with_plots.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set font configuration
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

def analyze_joint_variance(all_qpos_data):
    """Analyze joint variance"""
    if not all_qpos_data:
        return None
    
    # Convert to numpy array with shape: (num_episodes, episode_length, 14)
    all_qpos = np.array(all_qpos_data)
    
    num_episodes, episode_length, num_joints = all_qpos.shape
    print(f"Data shape: {num_episodes} episodes, {episode_length} timesteps, {num_joints} joints")
    
    # Calculate statistics for each timestep
    trajectory_stats = {
        'num_episodes': num_episodes,
        'episode_length': episode_length,
        'joints': {}
    }
    
    # Analyze each joint
    for joint_idx in range(num_joints):
        joint_data = all_qpos[:, :, joint_idx]  # shape: (num_episodes, episode_length)
        
        # Calculate statistics for each timestep
        mean_trajectory = np.mean(joint_data, axis=0)  # shape: (episode_length,)
        std_trajectory = np.std(joint_data, axis=0)    # shape: (episode_length,)
        var_trajectory = np.var(joint_data, axis=0)    # shape: (episode_length,)
        
        # Calculate confidence interval (mean ± std)
        upper_bound = mean_trajectory + std_trajectory
        lower_bound = mean_trajectory - std_trajectory
        
        # Calculate global statistics
        global_mean = np.mean(joint_data)
        global_std = np.std(joint_data)
        global_var = np.var(joint_data)
        
        # Calculate average variance across timesteps
        mean_variance = np.mean(var_trajectory)
        max_variance = np.max(var_trajectory)
        min_variance = np.min(var_trajectory)
        
        trajectory_stats['joints'][joint_idx] = {
            'mean': mean_trajectory,
            'std': std_trajectory,
            'var': var_trajectory,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'raw_data': joint_data,
            'global_mean': global_mean,
            'global_std': global_std,
            'global_var': global_var,
            'mean_variance': mean_variance,
            'max_variance': max_variance,
            'min_variance': min_variance
        }
    
    return trajectory_stats

def plot_joint_trajectories_with_variance(stats):
    """Plot trajectory charts for 14 joints, showing mean and variance regions"""
    if not stats:
        print("No data available for plotting")
        return
    
    # Bimanual ALOHA joint names
    joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
    
    # Create 14 subplots (7 left arm joints + 7 right arm joints)
    fig, axes = plt.subplots(7, 2, figsize=(20, 25))
    fig.suptitle('Slot Insertion Task - Joint Trajectory Analysis with Variance', fontsize=16)
    
    # Plot left arm joints (first column)
    for i, joint_name in enumerate(joint_names):
        ax = axes[i, 0]
        
        if i in stats['joints']:
            joint_data = stats['joints'][i]
            time_steps = np.arange(len(joint_data['mean']))
            
            # Plot mean trajectory
            ax.plot(time_steps, joint_data['mean'], 
                   color='blue', 
                   label='Mean trajectory', 
                   linewidth=2)
            
            # Plot variance region (mean ± std)
            ax.fill_between(time_steps, 
                          joint_data['lower_bound'], 
                          joint_data['upper_bound'], 
                          color='blue', 
                          alpha=0.3, 
                          label='±1σ variance')
            
            # Add statistical information to title
            mean_var = joint_data['mean_variance']
            ax.set_title(f'Left {joint_name} (Avg Var: {mean_var:.6f})')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Position (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot right arm joints (second column)
    for i, joint_name in enumerate(joint_names):
        ax = axes[i, 1]
        joint_idx = i + 7  # Right arm joint indices start from 7
        
        if joint_idx in stats['joints']:
            joint_data = stats['joints'][joint_idx]
            time_steps = np.arange(len(joint_data['mean']))
            
            # Plot mean trajectory
            ax.plot(time_steps, joint_data['mean'], 
                   color='red', 
                   label='Mean trajectory', 
                   linewidth=2)
            
            # Plot variance region (mean ± std)
            ax.fill_between(time_steps, 
                          joint_data['lower_bound'], 
                          joint_data['upper_bound'], 
                          color='red', 
                          alpha=0.3, 
                          label='±1σ variance')
            
            # Add statistical information to title
            mean_var = joint_data['mean_variance']
            ax.set_title(f'Right {joint_name} (Avg Var: {mean_var:.6f})')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Position (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('slot_insertion_joint_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_variance_over_time(stats):
    """Plot variance over time for each joint"""
    if not stats:
        return
    
    joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
    
    # Create variance plots
    fig, axes = plt.subplots(7, 2, figsize=(20, 25))
    fig.suptitle('Slot Insertion Task - Joint Variance Over Time', fontsize=16)
    
    # Left arm joint variance
    for i, joint_name in enumerate(joint_names):
        ax = axes[i, 0]
        
        if i in stats['joints']:
            joint_data = stats['joints'][i]
            time_steps = np.arange(len(joint_data['var']))
            
            ax.plot(time_steps, joint_data['var'], 
                   color='blue', 
                   linewidth=2)
            
            # Add mean variance line
            ax.axhline(y=joint_data['mean_variance'], 
                      color='blue', 
                      linestyle='--', 
                      alpha=0.7,
                      label=f'Mean Var: {joint_data["mean_variance"]:.6f}')
        
        ax.set_title(f'Left {joint_name} - Variance')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Right arm joint variance
    for i, joint_name in enumerate(joint_names):
        ax = axes[i, 1]
        joint_idx = i + 7
        
        if joint_idx in stats['joints']:
            joint_data = stats['joints'][joint_idx]
            time_steps = np.arange(len(joint_data['var']))
            
            ax.plot(time_steps, joint_data['var'], 
                   color='red', 
                   linewidth=2)
            
            # Add mean variance line
            ax.axhline(y=joint_data['mean_variance'], 
                      color='red', 
                      linestyle='--', 
                      alpha=0.7,
                      label=f'Mean Var: {joint_data["mean_variance"]:.6f}')
        
        ax.set_title(f'Right {joint_name} - Variance')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('slot_insertion_variance_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_variance_summary(stats):
    """Create variance analysis summary report"""
    if not stats:
        return
    
    print("\n" + "="*100)
    print("SLOT INSERTION TASK - JOINT VARIANCE ANALYSIS SUMMARY")
    print("="*100)
    
    print(f"Dataset: converted_bimanual_aloha_slot_insertion")
    print(f"Number of episodes: {stats['num_episodes']}")
    print(f"Episode length: {stats['episode_length']} timesteps")
    
    joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
    
    print(f"\n{'Joint':<15} {'Side':<6} {'Global Mean':<12} {'Global Std':<12} {'Mean Var':<12} {'Max Var':<12} {'Min Var':<12}")
    print("-" * 90)
    
    # Collect all variance data for sorting
    variance_data = []
    
    for i, joint_name in enumerate(joint_names):
        # Left arm
        if i in stats['joints']:
            joint_data = stats['joints'][i]
            variance_data.append({
                'joint': joint_name,
                'side': 'Left',
                'index': i,
                'global_mean': joint_data['global_mean'],
                'global_std': joint_data['global_std'],
                'mean_var': joint_data['mean_variance'],
                'max_var': joint_data['max_variance'],
                'min_var': joint_data['min_variance']
            })
        
        # Right arm
        joint_idx = i + 7
        if joint_idx in stats['joints']:
            joint_data = stats['joints'][joint_idx]
            variance_data.append({
                'joint': joint_name,
                'side': 'Right',
                'index': joint_idx,
                'global_mean': joint_data['global_mean'],
                'global_std': joint_data['global_std'],
                'mean_var': joint_data['mean_variance'],
                'max_var': joint_data['max_variance'],
                'min_var': joint_data['min_variance']
            })
    
    # Output sorted by average variance
    for data in variance_data:
        print(f"{data['joint']:<15} {data['side']:<6} {data['global_mean']:<12.6f} {data['global_std']:<12.6f} "
              f"{data['mean_var']:<12.6f} {data['max_var']:<12.6f} {data['min_var']:<12.6f}")
    
    # Find joints with highest and lowest variance
    print("\n" + "="*60)
    print("VARIANCE RANKING ANALYSIS")
    print("="*60)
    
    # Sort by average variance
    sorted_by_variance = sorted(variance_data, key=lambda x: x['mean_var'], reverse=True)
    
    print("\nHighest variance joints (Top 5):")
    for i, data in enumerate(sorted_by_variance[:5]):
        print(f"{i+1}. {data['side']} {data['joint']}: {data['mean_var']:.6f}")
    
    print("\nLowest variance joints (Bottom 5):")
    for i, data in enumerate(sorted_by_variance[-5:]):
        print(f"{i+1}. {data['side']} {data['joint']}: {data['mean_var']:.6f}")
    
    # Calculate left vs right arm variance comparison
    print("\n" + "="*60)
    print("LEFT VS RIGHT ARM VARIANCE COMPARISON")
    print("="*60)
    
    left_variances = [d['mean_var'] for d in variance_data if d['side'] == 'Left']
    right_variances = [d['mean_var'] for d in variance_data if d['side'] == 'Right']
    
    print(f"Left arm average variance: {np.mean(left_variances):.6f}")
    print(f"Right arm average variance: {np.mean(right_variances):.6f}")
    print(f"Variance ratio (Right/Left): {np.mean(right_variances)/np.mean(left_variances):.3f}")

def main():
    """Main function"""
    print("Starting joint variance analysis for converted_bimanual_aloha_slot_insertion dataset...")
    
    # Data path
    data_path = Path('/home/zzt/actnew/data/converted_bimanual_aloha_slot_insertion')
    
    # Get all episode files
    episode_files = sorted(list(data_path.glob('episode_*.hdf5')))
    
    if not episode_files:
        print(f"No episode files found in {data_path}.")
        return
        
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
        print("Failed to load any data.")
        return

    print(f"Successfully loaded data from {len(all_qpos_list)} episodes")
    
    # Analyze joint variance
    print("Analyzing joint variance...")
    stats = analyze_joint_variance(all_qpos_list)
    
    if not stats:
        print("Variance analysis failed.")
        return
    
    # Create summary statistics
    create_variance_summary(stats)
    
    # Plot trajectory charts
    print("\nPlotting joint trajectory charts...")
    plot_joint_trajectories_with_variance(stats)
    
    # Plot variance charts
    print("Plotting variance over time charts...")
    plot_variance_over_time(stats)
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("1. slot_insertion_joint_trajectories.png - Joint trajectory charts (with variance regions)")
    print("2. slot_insertion_variance_over_time.png - Variance over time charts")

if __name__ == "__main__":
    main()