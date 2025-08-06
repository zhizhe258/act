#!/usr/bin/env python3
"""
Episode Dataset for ACT Training
================================

返回完整episode数据的数据集类，用于ACT模型训练。
"""

import torch
import torch.utils.data as data
import h5py
import numpy as np
import os

class EpisodeDataset(data.Dataset):
    """返回完整episode数据的数据集"""
    
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = True  # 对于转换的数据，总是sim
    
    def __len__(self):
        return len(self.episode_ids)
    
    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        
        with h5py.File(dataset_path, 'r') as root:
            # 获取完整episode的数据
            qpos_full = root['/observations/qpos'][()]  # (episode_len, 14)
            qvel_full = root['/observations/qvel'][()]  # (episode_len, 14)
            action_full = root['/action'][()]  # (episode_len, 14)
            
            # 获取图像数据
            image_dict = {}
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]  # (episode_len, H, W, 3)
            
            episode_len = len(qpos_full)
            
            # 应用gripper归一化
            qpos_full = qpos_full.copy()
            # Left gripper (joint 6)
            mask_left = qpos_full[:, 6] > 0.4
            qpos_full[mask_left, 6] = (qpos_full[mask_left, 6] - 0.5) * 2.0
            qpos_full[:, 6] = np.clip(qpos_full[:, 6], 0.0, 1.0)
            # Right gripper (joint 13)
            mask_right = qpos_full[:, 13] > 0.4
            qpos_full[mask_right, 13] = (qpos_full[mask_right, 13] - 0.5) * 2.0
            qpos_full[:, 13] = np.clip(qpos_full[:, 13], 0.0, 1.0)
            
            # 归一化数据
            qpos_norm = (qpos_full - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']
            action_norm = (action_full - self.norm_stats['action_mean']) / self.norm_stats['action_std']
            
            # 处理图像数据
            all_cam_images = []
            for cam_name in self.camera_names:
                cam_images = image_dict[cam_name]  # (episode_len, H, W, 3)
                # 转换为CHW格式并归一化
                cam_images = np.transpose(cam_images, (0, 3, 1, 2))  # (episode_len, 3, H, W)
                cam_images = cam_images.astype(np.float32) / 255.0
                all_cam_images.append(cam_images)
            
            # 堆叠所有相机 (episode_len, num_cam, 3, H, W)
            images = np.stack(all_cam_images, axis=1)
            
            # 创建padding mask
            is_pad = np.zeros(episode_len, dtype=bool)
            
            # 转换为tensor
            qpos_tensor = torch.from_numpy(qpos_norm).float()
            action_tensor = torch.from_numpy(action_norm).float()
            images_tensor = torch.from_numpy(images).float()
            is_pad_tensor = torch.from_numpy(is_pad)
            
            return images_tensor, qpos_tensor, action_tensor, is_pad_tensor

def load_episode_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    """加载episode数据"""
    print(f'\nEpisode Data from: {dataset_dir}\n')
    
    # 划分训练和验证集
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    
    # 获取归一化统计信息
    from utils import get_norm_stats
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    
    # 创建数据集
    train_dataset = EpisodeDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodeDataset(val_indices, dataset_dir, camera_names, norm_stats)
    
    # 创建数据加载器
    train_dataloader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=1, 
        prefetch_factor=1
    )
    val_dataloader = data.DataLoader(
        val_dataset, 
        batch_size=batch_size_val, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=1, 
        prefetch_factor=1
    )
    
    return train_dataloader, val_dataloader, norm_stats, True 