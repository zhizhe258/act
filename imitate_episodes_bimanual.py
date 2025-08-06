#!/usr/bin/env python3
"""
Bimanual ALOHA Training Script - Optimized for Pure Training
============================================================

This script is specifically designed for training on bimanual ALOHA tasks
without unnecessary MuJoCo/XML dependencies during training.

Key improvements:
- No MuJoCo dependencies during training
- Proper bimanual task state dimension handling  
- Simplified data loading and preprocessing
- Clean separation between training and evaluation
"""

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

# Core imports (no MuJoCo dependencies)
from utils import load_data, compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy, CNNMLPPolicy
from auto_path_manager import create_organized_structure, print_structure_info

import IPython
e = IPython.embed

# Bimanual task configurations
BIMANUAL_TASK_CONFIGS = {
    'bimanual_aloha_cube_transfer': {
        'dataset_dir': '/home/zzt/actnew/data/bimanual_aloha_cube_transfer',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'state_dim': 14,
        'action_dim': 14,
    },
    'bimanual_aloha_peg_insertion': {
        'dataset_dir': '/home/zzt/actnew/data/bimanual_aloha_peg_insertion', 
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'state_dim': 14,
        'action_dim': 14,
    },
    'bimanual_aloha_slot_insertion': {
        'dataset_dir': '/home/zzt/actnew/data/bimanual_aloha_slot_insertion',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'state_dim': 14,
        'action_dim': 14,
    },
    'converted_bimanual_aloha_slot_insertion_with_vel': {
        'dataset_dir': '/home/zzt/actnew/data/converted_bimanual_aloha_slot_insertion_with_vel',  # 使用转换后的HDF5数据
        'num_episodes': 2,  # 使用转换的2个episodes，减少内存使用
        'episode_len': 300,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'state_dim': 14,
        'action_dim': 14,
    },
    'bimanual_aloha_color_cubes': {
        'dataset_dir': '/home/zzt/actnew/data/bimanual_aloha_color_cubes',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'state_dim': 14,
        'action_dim': 14,
    },
    'bimanual_aloha_hook_package': {
        'dataset_dir': '/home/zzt/actnew/data/bimanual_aloha_hook_package',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'state_dim': 14,
        'action_dim': 14,
    },
    'bimanual_aloha_pour_test_tube': {
        'dataset_dir': '/home/zzt/actnew/data/bimanual_aloha_pour_test_tube',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'state_dim': 14,
        'action_dim': 14,
    },
    'bimanual_aloha_thread_needle': {
        'dataset_dir': '/home/zzt/actnew/data/bimanual_aloha_thread_needle',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'state_dim': 14,
        'action_dim': 14,
    },
}

def get_task_config(task_name):
    """Get task configuration for bimanual ALOHA tasks"""
    if task_name in BIMANUAL_TASK_CONFIGS:
        return BIMANUAL_TASK_CONFIGS[task_name]
    else:
        raise ValueError(f"Unknown bimanual task: {task_name}")

def main(args):
    set_seed(1)
    
    # Extract arguments
    is_eval = args['eval']
    policy_class = args['policy_class']
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

    # Get task configuration
    task_config = get_task_config(task_name)
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    state_dim = task_config['state_dim']
    action_dim = task_config['action_dim']

    print(f"\n🤖 Bimanual ALOHA Training Configuration:")
    print(f"   Task: {task_name}")
    print(f"   State dimension: {state_dim}")
    print(f"   Action dimension: {action_dim}")
    print(f"   Dataset: {dataset_dir}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Episode length: {episode_len}")
    print(f"   Cameras: {camera_names}")

    # Configure policy parameters
    lr_backbone = 1e-5
    backbone = args.get('backbone', 'resnet18')
    
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            'lr': args['lr'],
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
    elif policy_class == 'RTC_Improved':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            'lr': args['lr'],
            'num_queries': args['chunk_size'],
            'freeze_steps': args.get('freeze_steps', args['chunk_size'] // 4),
            'kl_weight': args['kl_weight'],
            'hidden_dim': args['hidden_dim'],
            'dim_feedforward': args['dim_feedforward'],
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'nheads': nheads,
            'camera_names': camera_names,
            'dropout': args.get('dropout', 0.1),
            'pre_norm': args.get('pre_norm', False),
            'state_dim': state_dim,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'position_embedding': 'sine',
            'dilation': False,
            'weight_decay': 1e-4,
            'clip_max_norm': 0.1,
            'masks': False,
        }
    elif policy_class == 'CNNMLP':
        policy_config = {
            'lr': args['lr'], 
            'lr_backbone': lr_backbone, 
            'backbone': backbone, 
            'num_queries': 1,
            'camera_names': camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'dataset_dir': dataset_dir,
        'num_episodes': num_episodes,
    }

    if is_eval:
        print("❌ Evaluation mode requires environment interaction")
        print("   Use the original imitate_episodes.py for evaluation")
        print("   This script is optimized for training only")
        exit(1)

    # Load training data (HDF5 format)
    print(f"\n📁 Loading training data from: {dataset_dir}")
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val
    )

    # Validate data dimensions
    print(f"\n🔍 Validating data dimensions...")
    sample_batch = next(iter(train_dataloader))
    image_data, qpos_data, action_data, is_pad = sample_batch
    
    print(f"   Image data shape: {image_data.shape}")
    print(f"   QPos data shape: {qpos_data.shape}")
    print(f"   Action data shape: {action_data.shape}")
    print(f"   Expected state_dim: {state_dim}")
    print(f"   Expected action_dim: {action_dim}")
    
    # Verify dimensions match configuration
    if qpos_data.shape[-1] != state_dim:
        raise ValueError(f"QPos dimension mismatch: got {qpos_data.shape[-1]}, expected {state_dim}")
    if action_data.shape[-1] != action_dim:
        raise ValueError(f"Action dimension mismatch: got {action_data.shape[-1]}, expected {action_dim}")
    
    print("✅ Data dimensions validated")

    # Save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"💾 Dataset stats saved to: {stats_path}")

    # Train model
    print(f"\n🚀 Starting training...")
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # Save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'✅ Best checkpoint saved: {ckpt_path}')
    print(f'   Best validation loss: {min_val_loss:.6f} @ epoch {best_epoch}')

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
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    """Create optimizer for policy"""
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'RTC_Improved':
        optimizer = policy.optimizer
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def forward_pass(data, policy):
    """Forward pass through policy"""
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.cuda()
    qpos_data = qpos_data.cuda()
    action_data = action_data.cuda()
    is_pad = is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)

def train_bc(train_dataloader, val_dataloader, config):
    """Train behavior cloning policy"""
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    # Create policy
    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    
    # Enable RTC training if using RTC_Improved policy
    if policy_class == 'RTC_Improved' and hasattr(policy, 'enable_rtc_training'):
        policy.enable_rtc_training(True, rtc_weight=0.2)
        print("🔄 RTC training enabled with initial weight 0.2")
    
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    
    print(f"\n📊 Training for {num_epochs} epochs...")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        print(f'\n📈 Epoch {epoch}')
        
        # RTC progressive training schedule
        if policy_class == 'RTC_Improved' and hasattr(policy, 'progressive_training_schedule'):
            policy.progressive_training_schedule(epoch, num_epochs)
            if hasattr(policy, 'rtc_loss_weight'):
                print(f'   RTC loss weight: {policy.rtc_loss_weight:.3f}')
        
        # Validation
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
        
        print(f'   Val loss:   {epoch_val_loss:.5f}')
        summary_string = '   Val metrics: '
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # Training
        policy.train()
        optimizer.zero_grad()
        epoch_train_dicts = []
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # Backward pass
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_dicts.append(detach_dict(forward_dict))
            
        train_history.extend(epoch_train_dicts)
        epoch_summary = compute_dict_mean(epoch_train_dicts)
        epoch_train_loss = epoch_summary['loss']
        print(f'   Train loss: {epoch_train_loss:.5f}')
        summary_string = '   Train metrics: '
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # Save periodic checkpoints
        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    # Save final checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'✅ Training finished: Seed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # Save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    """Plot and save training curves"""
    if not train_history or not validation_history:
        return
        
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure(figsize=(10, 6))
        
        # Calculate training curve (average over batches within epochs)
        batches_per_epoch = len(train_history) // len(validation_history)
        train_epoch_values = []
        for epoch in range(len(validation_history)):
            start_idx = epoch * batches_per_epoch
            end_idx = min((epoch + 1) * batches_per_epoch, len(train_history))
            if start_idx < len(train_history):
                epoch_values = [train_history[i][key].item() for i in range(start_idx, end_idx)]
                train_epoch_values.append(np.mean(epoch_values))
        
        val_values = [summary[key].item() for summary in validation_history]
        
        epochs = np.arange(len(val_values))
        plt.plot(epochs, train_epoch_values[:len(val_values)], 'b-', label='train', linewidth=2)
        plt.plot(epochs, val_values, 'r-', label='validation', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.title(f'Training Curves - {key}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
    
    print(f'📊 Training curves saved to {ckpt_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bimanual ALOHA Training Script')
    
    # Core arguments
    parser.add_argument('--eval', action='store_true', help='Evaluation mode (not supported in this script)')
    parser.add_argument('--ckpt_dir', type=str, help='Checkpoint directory (optional, auto-generated if not provided)')
    parser.add_argument('--policy_class', type=str, required=True, 
                       choices=['ACT', 'RTC_Improved', 'CNNMLP'],
                       help='Policy class')
    parser.add_argument('--task_name', type=str, required=True,
                       choices=list(BIMANUAL_TASK_CONFIGS.keys()),
                       help='Bimanual task name')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')

    # Policy-specific arguments
    parser.add_argument('--kl_weight', type=int, default=10, help='KL Weight for ACT/RTC')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for ACT/RTC')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--dim_feedforward', type=int, default=3200, help='Feedforward dimension')
    parser.add_argument('--temporal_agg', action='store_true', help='Enable temporal aggregation')
    
    # RTC_Improved specific
    parser.add_argument('--freeze_steps', type=int, help='Freeze steps for RTC (default: chunk_size//4)')
    
    # Backbone selection
    parser.add_argument('--backbone', type=str, default='resnet18',
                       help='Backbone model (resnet18/34/50/101 or dinov2_vits14/vitb14/vitl14/vitg14)')
    
    args = vars(parser.parse_args())
    main(args)