#!/usr/bin/env python3
"""
Bimanual ALOHA Training Launcher
===============================

Simple training launcher with predefined configurations for bimanual ALOHA tasks.
This script provides easy-to-use training commands without needing MuJoCo dependencies.
"""

import os
import subprocess
import sys
import argparse

def get_training_configs():
    """Predefined training configurations for different tasks and policies"""
    return {
        'slot_insertion_act': {
            'task_name': 'converted_bimanual_aloha_slot_insertion_with_vel',
            'policy_class': 'ACT',
            'batch_size': 8,
            'seed': 0,
            'num_epochs': 2000,
            'lr': 1e-5,
            'kl_weight': 10,
            'chunk_size': 100,
            'hidden_dim': 512,
            'dim_feedforward': 3200,
            'backbone': 'resnet18',
        },
        'slot_insertion_rtc': {
            'task_name': 'converted_bimanual_aloha_slot_insertion_with_vel',
            'policy_class': 'RTC_Improved',
            'batch_size': 8,
            'seed': 0,
            'num_epochs': 2000,
            'lr': 1e-5,
            'kl_weight': 10,
            'chunk_size': 100,
            'hidden_dim': 512,
            'dim_feedforward': 3200,
            'backbone': 'resnet18',
            'freeze_steps': 25,  # chunk_size // 4
        },
        'cube_transfer_act': {
            'task_name': 'bimanual_aloha_cube_transfer',
            'policy_class': 'ACT',
            'batch_size': 8,
            'seed': 0,
            'num_epochs': 2000,
            'lr': 1e-5,
            'kl_weight': 10,
            'chunk_size': 100,
            'hidden_dim': 512,
            'dim_feedforward': 3200,
            'backbone': 'resnet18',
        },
        'peg_insertion_act': {
            'task_name': 'bimanual_aloha_peg_insertion',
            'policy_class': 'ACT',
            'batch_size': 8,
            'seed': 0,
            'num_epochs': 2000,
            'lr': 1e-5,
            'kl_weight': 10,
            'chunk_size': 100,
            'hidden_dim': 512,
            'dim_feedforward': 3200,
            'backbone': 'resnet18',
        },
    }

def build_command(config, temporal_agg=False):
    """Build training command from configuration"""
    cmd = [
        'python3', 'imitate_episodes_bimanual.py',
        '--task_name', config['task_name'],
        '--policy_class', config['policy_class'],
        '--batch_size', str(config['batch_size']),
        '--seed', str(config['seed']),
        '--num_epochs', str(config['num_epochs']),
        '--lr', str(config['lr']),
        '--kl_weight', str(config['kl_weight']),
        '--chunk_size', str(config['chunk_size']),
        '--hidden_dim', str(config['hidden_dim']),
        '--dim_feedforward', str(config['dim_feedforward']),
        '--backbone', config['backbone'],
    ]
    
    if temporal_agg:
        cmd.append('--temporal_agg')
    
    if 'freeze_steps' in config:
        cmd.extend(['--freeze_steps', str(config['freeze_steps'])])
    
    return cmd

def main():
    parser = argparse.ArgumentParser(description='Bimanual ALOHA Training Launcher')
    parser.add_argument('--config', type=str, required=True,
                       choices=list(get_training_configs().keys()),
                       help='Training configuration name')
    parser.add_argument('--temporal_agg', action='store_true',
                       help='Enable temporal aggregation')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print command without executing')
    parser.add_argument('--seed', type=int,
                       help='Override seed (optional)')
    parser.add_argument('--batch_size', type=int,
                       help='Override batch size (optional)')
    parser.add_argument('--num_epochs', type=int,
                       help='Override number of epochs (optional)')
    
    args = parser.parse_args()
    
    # Get configuration
    configs = get_training_configs()
    config = configs[args.config].copy()
    
    # Override parameters if provided
    if args.seed is not None:
        config['seed'] = args.seed
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    
    # Build command
    cmd = build_command(config, args.temporal_agg)
    
    print(f"🤖 Bimanual ALOHA Training")
    print(f"📝 Configuration: {args.config}")
    print(f"🎯 Task: {config['task_name']}")
    print(f"🧠 Policy: {config['policy_class']}")
    print(f"🔄 Temporal Aggregation: {args.temporal_agg}")
    print(f"🎲 Seed: {config['seed']}")
    print(f"📦 Batch Size: {config['batch_size']}")
    print(f"📊 Epochs: {config['num_epochs']}")
    print()
    
    print("Command to execute:")
    print(" ".join(cmd))
    print()
    
    if args.dry_run:
        print("🔍 Dry run mode - command not executed")
        return
    
    # Execute training
    print("🚀 Starting training...")
    try:
        subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main()