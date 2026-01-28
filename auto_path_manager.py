"""
Auto Path Manager for ACT Project
Automatically creates organized directory structures based on task names and data types
"""

import os
from datetime import datetime


def get_auto_dataset_dir(task_name, base_dir="./data", timestamp=False):
    """
    Auto-generate dataset directory based on task name
    Args:
        task_name: Name of the task (e.g., 'sim_transfer_cube_scripted')
        base_dir: Base directory for all data (default: './data')
        timestamp: Whether to add timestamp suffix
    Returns:
        dataset_dir: Full path to the dataset directory
    """
    if timestamp:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = os.path.join(base_dir, f"{task_name}_{time_str}")
    else:
        dataset_dir = os.path.join(base_dir, task_name)
    
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir


def get_auto_ckpt_dir(task_name, policy_class, base_dir="./checkpoints", timestamp=False, 
                      chunk_config=None, no_cvae=False):
    """
    Auto-generate checkpoint directory based on task name, policy, and configurations
    Args:
        task_name: Name of the task
        policy_class: Policy class name (e.g., 'ACT', 'CNNMLP')
        base_dir: Base directory for all checkpoints
        timestamp: Whether to add timestamp suffix
        chunk_config: Dict with chunk sampling parameters (optional)
        no_cvae: Whether using pure BC mode without CVAE (adds _nocvae suffix)
    Returns:
        ckpt_dir: Full path to the checkpoint directory
    """
    # Build base directory name
    dir_name_parts = [task_name, policy_class]
    
    # Add nocvae suffix for pure BC mode
    if no_cvae and policy_class == 'ACT':
        dir_name_parts.append("nocvae")
    
    # Add chunk size to directory name
    if chunk_config and chunk_config.get('chunk_size'):
        chunk_size = chunk_config.get('chunk_size', 100)
        dir_name_parts.append(f"chunk_{chunk_size}")
    
    # Add timestamp (if needed)
    if timestamp:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name_parts.append(time_str)
    
    # Combine final directory name
    dir_name = "_".join(dir_name_parts)
    ckpt_dir = os.path.join(base_dir, dir_name)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def get_auto_eval_dir(ckpt_dir, eval_type="evaluation"):
    """
    Auto-generate evaluation results directory
    Args:
        ckpt_dir: Checkpoint directory path
        eval_type: Type of evaluation (e.g., 'evaluation', 'trajectory_comparison')
    Returns:
        eval_dir: Full path to the evaluation directory
    """
    eval_dir = os.path.join(ckpt_dir, eval_type)
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def create_organized_structure(task_name, policy_class=None, base_data_dir="./data", base_ckpt_dir="./checkpoints",
                              chunk_config=None, no_cvae=False):
    """
    Create complete organized directory structure for a task
    Args:
        task_name: Name of the task
        policy_class: Policy class name (optional, for training/eval)
        base_data_dir: Base directory for datasets
        base_ckpt_dir: Base directory for checkpoints
        chunk_config: Dict with chunk sampling parameters (optional)
        no_cvae: Whether using pure BC mode without CVAE
    Returns:
        dict: Dictionary containing all created paths
    """
    paths = {}
    
    # Dataset directory
    paths['dataset_dir'] = get_auto_dataset_dir(task_name, base_data_dir)
    
    # Checkpoint directory (if policy specified)
    if policy_class:
        paths['ckpt_dir'] = get_auto_ckpt_dir(task_name, policy_class, base_ckpt_dir, 
                                            chunk_config=chunk_config, no_cvae=no_cvae)
        paths['eval_dir'] = get_auto_eval_dir(paths['ckpt_dir'])
        paths['trajectory_dir'] = get_auto_eval_dir(paths['ckpt_dir'], 'trajectory_comparisons')
    
    return paths


def print_structure_info(paths):
    """
    Print information about created directory structure
    """
    print("\n" + "="*60)
    print("AUTO-CREATED DIRECTORY STRUCTURE")
    print("="*60)
    for key, path in paths.items():
        print(f"{key:20}: {path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    task_name = "sim_transfer_cube_scripted"
    policy_class = "ACT"
    
    paths = create_organized_structure(task_name, policy_class)
    print_structure_info(paths)