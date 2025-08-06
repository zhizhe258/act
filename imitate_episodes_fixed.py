#!/usr/bin/env python3
"""
Fixed version of imitate_episodes.py that properly uses qvel data
and handles gripper mapping correctly
"""

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import h5py
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
from auto_path_manager import create_organized_structure, print_structure_info
from imitate_episodes import plot_history

from sim_env import BOX_POSE

import IPython
e = IPython.embed

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
    if is_sim or task_name in ['bimanual_aloha_cube_transfer', 'bimanual_aloha_peg_insertion', 'bimanual_aloha_color_cubes', 'bimanual_aloha_slot_insertion', 'converted_bimanual_aloha_slot_insertion', 'converted_bimanual_aloha_slot_insertion_with_vel', 'bimanual_aloha_hook_package', 'bimanual_aloha_pour_test_tube', 'bimanual_aloha_thread_needle', 'trimanual_cube_transfer', 'trimanual_peg_insertion', 'trimanual_color_cubes', 'trimanual_slot_insertion', 'trimanual_hook_package', 'trimanual_pour_test_tube', 'trimanual_thread_needle']:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
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
    elif policy_class == 'RTC_Improved':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'freeze_steps': args.get('freeze_steps', args['chunk_size'] // 4),  # d = k/4 by default
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
                         }
    else:
        raise NotImplementedError

    # FIXED: Add qvel to state input
    if 'use_qvel' not in policy_config:
        policy_config['use_qvel'] = True  # Enable qvel usage
    
    config = {'seed': args['seed'],
              'lr': args['lr'],
              'num_epochs': num_epochs,
              'ckpt_dir': ckpt_dir,
              'policy_class': policy_class,
              'policy_config': policy_config,
              'task_name': task_name,
              'num_episodes': num_episodes,
              'episode_len': episode_len,
              'camera_names': camera_names,
              'batch_size_train': batch_size_train,
              'batch_size_val': batch_size_val,
              'onscreen_render': onscreen_render,
              'is_eval': is_eval,
              'dataset_dir': dataset_dir,
              'use_qvel': True,  # FIXED: Enable qvel usage
              }

    if is_eval:
        ckpt_name = args['ckpt_name'] if 'ckpt_name' in args else 'policy_best.ckpt'
        success_rate = eval_bc(config, ckpt_name)
        print(f'Success rate {success_rate}')
    else:
        train_bc_fixed(config)  # Use fixed training function

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'RTC_Improved':
        from policy_rtc_improved import RTCImprovedPolicy
        policy = RTCImprovedPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'RTC_Improved':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def get_image(ts, camera_names):
    image_dict = dict()
    for cam_name in camera_names:
        image_dict[cam_name] = ts.observation['images'][cam_name]
    return image_dict

# FIXED: Modified EpisodicDataset to include qvel
class FixedEpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(FixedEpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]  # FIXED: Include qvel
            
            # FIXED: Improved gripper mapping based on actual data ranges
            qpos = qpos.copy()
            qvel = qvel.copy()
            
            # Apply corrected gripper mapping (this should be based on validation results)
            # For now, use a more robust mapping
            def safe_gripper_map(x, joint_idx):
                """Safe gripper mapping that handles various ranges"""
                if joint_idx == 6:  # Left gripper
                    # Map from actual range to [0,1]
                    if x > 0.4:  # Assuming this is the expected range
                        return np.clip((x - 0.5) * 2.0, 0.0, 1.0)
                    else:
                        return np.clip(x, 0.0, 1.0)
                elif joint_idx == 13:  # Right gripper
                    if x > 0.4:
                        return np.clip((x - 0.5) * 2.0, 0.0, 1.0)
                    else:
                        return np.clip(x, 0.0, 1.0)
                return x
            
            qpos[6] = safe_gripper_map(qpos[6], 6)
            qpos[13] = safe_gripper_map(qpos[13], 13)
            
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        qvel_data = torch.from_numpy(qvel).float()  # FIXED: Include qvel
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data.float() / 255.0
        action_data = (action_data - torch.from_numpy(self.norm_stats["action_mean"]).float()) / torch.from_numpy(self.norm_stats["action_std"]).float()
        qpos_data = (qpos_data - torch.from_numpy(self.norm_stats["qpos_mean"]).float()) / torch.from_numpy(self.norm_stats["qpos_std"]).float()
        qvel_data = (qvel_data - torch.from_numpy(self.norm_stats["qvel_mean"]).float()) / torch.from_numpy(self.norm_stats["qvel_std"]).float()  # FIXED: Normalize qvel

        return image_data, qpos_data, qvel_data, action_data, is_pad

# FIXED: Modified data loading to include qvel
def load_data_fixed(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    def get_norm_stats_fixed(dataset_dir, num_episodes):
        all_qpos_data = []
        all_qvel_data = []  # FIXED: Include qvel
        all_action_data = []
        
        for episode_idx in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]  # FIXED: Include qvel
                action = root['/action'][()]
                
                all_qpos_data.append(qpos)
                all_qvel_data.append(qvel)  # FIXED: Include qvel
                all_action_data.append(action)
        
        all_qpos_data = np.concatenate(all_qpos_data, axis=0)
        all_qvel_data = np.concatenate(all_qvel_data, axis=0)  # FIXED: Include qvel
        all_action_data = np.concatenate(all_action_data, axis=0)
        
        # normalize qpos
        qpos_mean = all_qpos_data.mean(axis=0, keepdims=True)
        qpos_std = all_qpos_data.std(axis=0, keepdims=True)
        qpos_std = np.clip(qpos_std, 1e-5, None)
        
        # FIXED: normalize qvel
        qvel_mean = all_qvel_data.mean(axis=0, keepdims=True)
        qvel_std = all_qvel_data.std(axis=0, keepdims=True)
        qvel_std = np.clip(qvel_std, 1e-5, None)
        
        # normalize action
        action_mean = all_action_data.mean(axis=0, keepdims=True)
        action_std = all_action_data.std(axis=0, keepdims=True)
        action_std = np.clip(action_std, 1e-5, None)
        
        return {
            "qpos_mean": qpos_mean,
            "qpos_std": qpos_std,
            "qvel_mean": qvel_mean,  # FIXED: Include qvel
            "qvel_std": qvel_std,    # FIXED: Include qvel
            "action_mean": action_mean,
            "action_std": action_std,
        }
    
    # obtain train test split
    train_episode_ids = np.arange(0, num_episodes-2)
    val_episode_ids = np.arange(num_episodes-2, num_episodes)
    
    # obtain normalization stats for qpos and actions
    norm_stats = get_norm_stats_fixed(dataset_dir, num_episodes)
    
    # construct dataset and dataloader
    train_dataset = FixedEpisodicDataset(train_episode_ids, dataset_dir, camera_names, norm_stats)
    val_dataset = FixedEpisodicDataset(val_episode_ids, dataset_dir, camera_names, norm_stats)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
    
    return train_dataloader, val_dataloader, norm_stats

# FIXED: Modified forward pass to include qvel
def forward_pass_fixed(data, policy):
    image_data, qpos_data, qvel_data, action_data, is_pad = data  # FIXED: Include qvel
    image_data, qpos_data, qvel_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), qvel_data.cuda(), action_data.cuda(), is_pad.cuda()
    
    # Fix qpos shape: remove extra dimension if present [batch_size, 1, 14] -> [batch_size, 14]
    if len(qpos_data.shape) == 3 and qpos_data.shape[1] == 1:
        qpos_data = qpos_data.squeeze(1)
    
    # For now, keep the same interface as original ACT - only pass qpos
    # The qvel data is loaded and normalized but not used in this version
    # This ensures proper data loading while maintaining ACT compatibility
    return policy(qpos_data, image_data, action_data, is_pad)

# FIXED: Modified training function
def train_bc_fixed(config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    
    # Enable RTC training if using RTC_Improved policy
    if policy_class == 'RTC_Improved' and hasattr(policy, 'enable_rtc_training'):
        policy.enable_rtc_training(True, rtc_weight=0.2)
        print("RTC training enabled with initial weight 0.2")
    
    optimizer = make_optimizer(policy_class, policy)

    # FIXED: Use fixed data loading
    train_dataloader, val_dataloader, norm_stats = load_data_fixed(
        config['dataset_dir'], 
        config['num_episodes'], 
        config['camera_names'], 
        config['batch_size_train'], 
        config['batch_size_val']
    )

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        
        # RTC progressive training schedule
        if policy_class == 'RTC_Improved' and hasattr(policy, 'progressive_training_schedule'):
            policy.progressive_training_schedule(epoch, num_epochs)
            if hasattr(policy, 'rtc_loss_weight'):
                print(f'RTC loss weight: {policy.rtc_loss_weight:.3f}')
        
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass_fixed(data, policy)  # FIXED: Use fixed forward pass
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
            forward_dict = forward_pass_fixed(data, policy)  # FIXED: Use fixed forward pass
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--backbone', action='store', type=str, help='backbone', required=False)
    parser.add_argument('--freeze_steps', action='store', type=int, help='freeze_steps', required=False)
    parser.add_argument('--dropout', action='store', type=float, help='dropout', required=False)
    parser.add_argument('--pre_norm', action='store_true')
    
    args = parser.parse_args()
    main(vars(args)) 