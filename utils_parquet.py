#!/usr/bin/env python3
"""
Parquet Data Loading Utils for Gym-ALOHA Datasets
================================================

专门处理parquet格式的gym-aloha数据集的加载工具
"""

import numpy as np
import torch
import pandas as pd
import os
import cv2
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
import io

def load_parquet_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    """
    加载parquet格式的gym-aloha数据集
    
    Args:
        dataset_dir: 数据集目录路径
        num_episodes: episode数量
        camera_names: 相机名称列表
        batch_size_train: 训练批大小
        batch_size_val: 验证批大小
    
    Returns:
        train_dataloader, val_dataloader, norm_stats, is_sim
    """
    print(f"📊 Loading parquet data from: {dataset_dir}")
    
    # 读取parquet文件
    parquet_path = os.path.join(dataset_dir, 'data', 'train-00000-of-00001.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"   Total data points: {len(df)}")
    print(f"   Episodes available: {df['episode_index'].max() + 1}")
    
    # 限制episode数量
    df = df[df['episode_index'] < num_episodes]
    print(f"   Using episodes: 0-{num_episodes-1}")
    
    # 获取数据统计信息用于归一化
    norm_stats = get_parquet_norm_stats(df)
    
    # 划分训练和验证集
    train_ratio = 0.8
    all_episodes = list(range(num_episodes))
    np.random.shuffle(all_episodes)
    train_episodes = all_episodes[:int(train_ratio * num_episodes)]
    val_episodes = all_episodes[int(train_ratio * num_episodes):]
    
    print(f"   Train episodes: {len(train_episodes)}")
    print(f"   Val episodes: {len(val_episodes)}")
    
    # 创建数据集
    train_dataset = ParquetEpisodicDataset(df, train_episodes, camera_names, norm_stats)
    val_dataset = ParquetEpisodicDataset(df, val_episodes, camera_names, norm_stats)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=1, 
        prefetch_factor=1
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size_val, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=1, 
        prefetch_factor=1
    )
    
    return train_dataloader, val_dataloader, norm_stats, True  # is_sim=True

def get_parquet_norm_stats(df):
    """计算parquet数据的归一化统计信息"""
    print("📈 Computing normalization statistics...")
    
    # 提取state和action数据
    all_states = []
    all_actions = []
    
    for episode_idx in df['episode_index'].unique():
        episode_data = df[df['episode_index'] == episode_idx].sort_values('frame_index')
        
        # 提取state数据 (14D)
        states = np.stack([np.frombuffer(state, dtype=np.float32) for state in episode_data['observation.state']])
        # 提取action数据 (14D)  
        actions = np.stack([np.frombuffer(action, dtype=np.float32) for action in episode_data['action']])
        
        all_states.append(states)
        all_actions.append(actions)
    
    # 合并所有数据
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"   State shape: {all_states.shape}")
    print(f"   Action shape: {all_actions.shape}")
    
    # 计算统计信息
    qpos_mean = all_states.mean(axis=0)
    qpos_std = all_states.std(axis=0)
    qpos_std = np.clip(qpos_std, 1e-6, None)  # 避免除零
    
    action_mean = all_actions.mean(axis=0)
    action_std = all_actions.std(axis=0)  
    action_std = np.clip(action_std, 1e-6, None)  # 避免除零
    
    # 打印gripper统计信息
    print(f"   Left gripper (dim 6): mean={qpos_mean[6]:.3f}, std={qpos_std[6]:.3f}")
    print(f"   Right gripper (dim 13): mean={qpos_mean[13]:.3f}, std={qpos_std[13]:.3f}")
    
    norm_stats = {
        'qpos_mean': qpos_mean,
        'qpos_std': qpos_std,
        'action_mean': action_mean,
        'action_std': action_std,
    }
    
    return norm_stats

class ParquetEpisodicDataset(Dataset):
    """Parquet格式的Episode数据集 - 返回完整episode"""
    
    def __init__(self, df, episode_indices, camera_names, norm_stats):
        self.df = df
        self.episode_indices = episode_indices
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        
        # 为每个episode创建索引
        self.episode_data = {}
        for episode_idx in episode_indices:
            episode_df = df[df['episode_index'] == episode_idx].sort_values('frame_index')
            self.episode_data[episode_idx] = episode_df
        
        print(f"📦 Dataset created with {len(self.episode_indices)} episodes")
    
    def __len__(self):
        return len(self.episode_indices)
    
    def __getitem__(self, idx):
        """返回完整episode的数据"""
        episode_idx = self.episode_indices[idx]
        episode_df = self.episode_data[episode_idx]
        
        # 提取整个episode的state和action数据
        episode_states = []
        episode_actions = []
        
        for _, row in episode_df.iterrows():
            state = np.frombuffer(row['observation.state'], dtype=np.float32)
            action = np.frombuffer(row['action'], dtype=np.float32)
            episode_states.append(state)
            episode_actions.append(action)
        
        # 转换为numpy数组
        episode_states = np.array(episode_states)  # (episode_len, 14)
        episode_actions = np.array(episode_actions)  # (episode_len, 14)
        
        # 限制到300步（如果episode更长）
        max_len = 300
        if len(episode_states) > max_len:
            episode_states = episode_states[:max_len]
            episode_actions = episode_actions[:max_len]
        
        # 归一化
        qpos = (episode_states - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']
        action = (episode_actions - self.norm_stats['action_mean']) / self.norm_stats['action_std']
        
        # 创建图像数据（暂时使用假数据）
        episode_len = len(qpos)
        all_cam_images = []
        
        for cam_name in self.camera_names:
            # 为每一帧创建假图像
            cam_images = []
            for t in range(episode_len):
                # 创建与时间步相关的假图像
                img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) * 0.1
                img = rearrange(img, 'h w c -> c h w')
                cam_images.append(img / 255.0)
            
            cam_images = np.stack(cam_images)  # (episode_len, 3, 480, 640)
            all_cam_images.append(cam_images)
        
        # 堆叠所有相机 (num_cameras, episode_len, 3, 480, 640)
        images = np.stack(all_cam_images, axis=0)
        
        # 创建padding mask
        is_pad = np.zeros(episode_len, dtype=bool)
        
        # 转换为tensor
        image = torch.from_numpy(images).float()
        qpos = torch.from_numpy(qpos).float()
        action = torch.from_numpy(action).float()
        is_pad = torch.from_numpy(is_pad)
        
        return image, qpos, action, is_pad
    
    def _extract_frame_from_video(self, video_path, frame_idx):
        """从视频文件中提取指定帧"""
        try:
            # 检查视频缓存
            if video_path not in self._video_cache:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return None
                self._video_cache[video_path] = cap
            else:
                cap = self._video_cache[video_path]
            
            # 设置到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                return frame
            else:
                return None
                
        except Exception as e:
            print(f"Warning: Failed to extract frame {frame_idx} from {video_path}: {e}")
            return None