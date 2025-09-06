import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import qmc

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
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
            qvel = root['/observations/qvel'][start_ts]
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
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

# Pose sampling configuration
POSE_SAMPLING_MODE = 'random'  # 'fixed', 'random', 'edge', 'similar', 'uniform'

# Fixed poses
FIXED_CUBE_POSE = [0.1, 0.5, 0.05, 1, 0, 0, 0]
FIXED_PEG_POSE = [0.15, 0.5, 0.05, 1, 0, 0, 0]
FIXED_SOCKET_POSE = [-0.15, 0.5, 0.05, 1, 0, 0, 0]

# Sampling ranges
CUBE_RANGES = {'x': [0.0, 0.2], 'y': [0.4, 0.6], 'z': [0.05, 0.05]}
PEG_RANGES = {'x': [0.1, 0.2], 'y': [0.4, 0.6], 'z': [0.05, 0.05]}
SOCKET_RANGES = {'x': [-0.2, -0.1], 'y': [0.4, 0.6], 'z': [0.05, 0.05]}

# Similar mode: small ranges around fixed positions for fine-tuning tasks
SIMILAR_PEG_RANGES = {'x': [0.16, 0.19], 'y': [0.48, 0.52], 'z': [0.05, 0.05]}  # 2cm x 2cm range around center
SIMILAR_SOCKET_RANGES = {'x': [-0.19, -0.16], 'y': [0.48, 0.52], 'z': [0.05, 0.05]}  # 2cm x 2cm range around center

# Global counters for LHS sampling
_box_pose_counter = 0
_peg_pose_counter = 0

def sample_box_pose():
    global _box_pose_counter
    
    if POSE_SAMPLING_MODE == 'fixed':
        return np.array(FIXED_CUBE_POSE)
    
    elif POSE_SAMPLING_MODE == 'uniform':
        # Uniform mode: use Latin Hypercube Sampling (LHS) for uniform distribution
        # Create LHS sampler for 3D space
        sampler = qmc.LatinHypercube(d=3, seed=42)
        # Generate samples in [0,1]^3
        samples = sampler.random(n=1000)
        # Scale to actual ranges
        x_range = CUBE_RANGES['x']
        y_range = CUBE_RANGES['y']
        z_range = CUBE_RANGES['z']
        
        # Get current sample based on counter
        sample_idx = _box_pose_counter % 1000
        x = samples[sample_idx, 0] * (x_range[1] - x_range[0]) + x_range[0]
        y = samples[sample_idx, 1] * (y_range[1] - y_range[0]) + y_range[0]
        z = samples[sample_idx, 2] * (z_range[1] - z_range[0]) + z_range[0]
        
        _box_pose_counter += 1
        cube_position = np.array([x, y, z])
    
    else:  # random
        x = np.random.uniform(CUBE_RANGES['x'][0], CUBE_RANGES['x'][1])
        y = np.random.uniform(CUBE_RANGES['y'][0], CUBE_RANGES['y'][1])
        z = np.random.uniform(CUBE_RANGES['z'][0], CUBE_RANGES['z'][1])
        cube_position = np.array([x, y, z])
    
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    global _peg_pose_counter
    
    if POSE_SAMPLING_MODE == 'fixed':
        peg_pose = np.array(FIXED_PEG_POSE)
        socket_pose = np.array(FIXED_SOCKET_POSE)
        return peg_pose, socket_pose
    
    elif POSE_SAMPLING_MODE == 'uniform':
        # Uniform mode: use Latin Hypercube Sampling (LHS) for uniform distribution
        # Create LHS sampler for 4D space (peg_x, peg_y, socket_x, socket_y)
        sampler = qmc.LatinHypercube(d=4, seed=42)
        # Generate samples in [0,1]^4
        samples = sampler.random(n=1000)
        
        # Get current sample based on counter
        sample_idx = _peg_pose_counter % 1000
        
        # Peg position: LHS sampling
        peg_x = samples[sample_idx, 0] * (PEG_RANGES['x'][1] - PEG_RANGES['x'][0]) + PEG_RANGES['x'][0]
        peg_y = samples[sample_idx, 1] * (PEG_RANGES['y'][1] - PEG_RANGES['y'][0]) + PEG_RANGES['y'][0]
        peg_position = np.array([peg_x, peg_y, PEG_RANGES['z'][0]])
        
        # Socket position: LHS sampling
        socket_x = samples[sample_idx, 2] * (SOCKET_RANGES['x'][1] - SOCKET_RANGES['x'][0]) + SOCKET_RANGES['x'][0]
        socket_y = samples[sample_idx, 3] * (SOCKET_RANGES['y'][1] - SOCKET_RANGES['y'][0]) + SOCKET_RANGES['y'][0]
        socket_position = np.array([socket_x, socket_y, SOCKET_RANGES['z'][0]])
        
        _peg_pose_counter += 1
    
    elif POSE_SAMPLING_MODE == 'edge':
        # Edge mode: peg and socket only appear at the edge positions of their respective Y-axis ranges
        # Use LHS sampling for X-axis distribution, Y-axis randomly distributed along the edges
        
        # Create LHS sampler for 2D space (peg_x, socket_x)
        sampler = qmc.LatinHypercube(d=2, seed=42)
        samples = sampler.random(n=1000)
        
        # Get current sample based on counter
        sample_idx = _peg_pose_counter % 1000
        
        # Peg position: X coordinate LHS distribution, Y coordinate randomly along edges
        peg_x = samples[sample_idx, 0] * (PEG_RANGES['x'][1] - PEG_RANGES['x'][0]) + PEG_RANGES['x'][0]
        # Randomly choose between lower edge (0.02 range) and upper edge (0.02 range)
        if np.random.random() < 0.5:
            peg_y = np.random.uniform(PEG_RANGES['y'][0], PEG_RANGES['y'][0] + 0.02)  # Lower edge
        else:
            peg_y = np.random.uniform(PEG_RANGES['y'][1] - 0.02, PEG_RANGES['y'][1])  # Upper edge
        peg_position = np.array([peg_x, peg_y, PEG_RANGES['z'][0]])
        
        # Socket position: X coordinate LHS distribution, Y coordinate randomly along edges
        socket_x = samples[sample_idx, 1] * (SOCKET_RANGES['x'][1] - SOCKET_RANGES['x'][0]) + SOCKET_RANGES['x'][0]
        # Randomly choose between lower edge (0.02 range) and upper edge (0.02 range)
        if np.random.random() < 0.5:
            socket_y = np.random.uniform(SOCKET_RANGES['y'][0], SOCKET_RANGES['y'][0] + 0.02)  # Lower edge
        else:
            socket_y = np.random.uniform(SOCKET_RANGES['y'][1] - 0.02, SOCKET_RANGES['y'][1])  # Upper edge
        socket_position = np.array([socket_x, socket_y, SOCKET_RANGES['z'][0]])
        
        _peg_pose_counter += 1
    
    elif POSE_SAMPLING_MODE == 'similar':
        # Similar mode: peg and socket move in very small ranges (fine-tuning tasks)
        # Use LHS sampling for small range distribution
        
        # Create LHS sampler for 4D space (peg_x, peg_y, socket_x, socket_y)
        sampler = qmc.LatinHypercube(d=4, seed=42)
        samples = sampler.random(n=1000)
        
        # Get current sample based on counter
        sample_idx = _peg_pose_counter % 1000
        
        # Peg position: LHS distribution in small range
        peg_x = samples[sample_idx, 0] * (SIMILAR_PEG_RANGES['x'][1] - SIMILAR_PEG_RANGES['x'][0]) + SIMILAR_PEG_RANGES['x'][0]
        peg_y = samples[sample_idx, 1] * (SIMILAR_PEG_RANGES['y'][1] - SIMILAR_PEG_RANGES['y'][0]) + SIMILAR_PEG_RANGES['y'][0]
        peg_position = np.array([peg_x, peg_y, SIMILAR_PEG_RANGES['z'][0]])
        
        # Socket position: LHS distribution in small range
        socket_x = samples[sample_idx, 2] * (SIMILAR_SOCKET_RANGES['x'][1] - SIMILAR_SOCKET_RANGES['x'][0]) + SIMILAR_SOCKET_RANGES['x'][0]
        socket_y = samples[sample_idx, 3] * (SIMILAR_SOCKET_RANGES['y'][1] - SIMILAR_SOCKET_RANGES['y'][0]) + SIMILAR_SOCKET_RANGES['y'][0]
        socket_position = np.array([socket_x, socket_y, SIMILAR_SOCKET_RANGES['z'][0]])
        
        _peg_pose_counter += 1
    
    else:  # random
        # Peg random sampling
        peg_x = np.random.uniform(PEG_RANGES['x'][0], PEG_RANGES['x'][1])
        peg_y = np.random.uniform(PEG_RANGES['y'][0], PEG_RANGES['y'][1])
        peg_z = np.random.uniform(PEG_RANGES['z'][0], PEG_RANGES['z'][1])
        peg_position = np.array([peg_x, peg_y, peg_z])
        
        # Socket random sampling
        socket_x = np.random.uniform(SOCKET_RANGES['x'][0], SOCKET_RANGES['x'][1])
        socket_y = np.random.uniform(SOCKET_RANGES['y'][0], SOCKET_RANGES['y'][1])
        socket_z = np.random.uniform(SOCKET_RANGES['z'][0], SOCKET_RANGES['z'][1])
        socket_position = np.array([socket_x, socket_y, socket_z])
    
    peg_quat = np.array([1, 0, 0, 0])
    socket_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])
    socket_pose = np.concatenate([socket_position, socket_quat])
    
    return peg_pose, socket_pose


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# Functions for forced random sampling during evaluation
def sample_box_pose_random():
    """Force random sampling during evaluation, unaffected by POSE_SAMPLING_MODE"""
    x = np.random.uniform(CUBE_RANGES['x'][0], CUBE_RANGES['x'][1])
    y = np.random.uniform(CUBE_RANGES['y'][0], CUBE_RANGES['y'][1])
    z = np.random.uniform(CUBE_RANGES['z'][0], CUBE_RANGES['z'][1])
    cube_position = np.array([x, y, z])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose_random():
    """Force random sampling during evaluation, unaffected by POSE_SAMPLING_MODE"""
    # Peg random sampling
    peg_x = np.random.uniform(PEG_RANGES['x'][0], PEG_RANGES['x'][1])
    peg_y = np.random.uniform(PEG_RANGES['y'][0], PEG_RANGES['y'][1])
    peg_z = np.random.uniform(PEG_RANGES['z'][0], PEG_RANGES['z'][1])
    peg_position = np.array([peg_x, peg_y, peg_z])
    
    # Socket random sampling
    socket_x = np.random.uniform(SOCKET_RANGES['x'][0], SOCKET_RANGES['x'][1])
    socket_y = np.random.uniform(SOCKET_RANGES['y'][0], SOCKET_RANGES['y'][1])
    socket_z = np.random.uniform(SOCKET_RANGES['z'][0], SOCKET_RANGES['z'][1])
    socket_position = np.array([socket_x, socket_y, socket_z])
    
    peg_quat = np.array([1, 0, 0, 0])
    socket_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])
    socket_pose = np.concatenate([socket_position, socket_quat])
  
    
    
    return peg_pose, socket_pose



# Cupboard
def sample_box_cupboard_pose():
    # box long
    x_range = [0.15, 0.20]  # 0.15 -- 0.20
    y_range = [0.45, 0.55]  # 0.45 -- 0.55
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    box_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    box_quat = np.array([1, 0, 0, 0])
    box_pose = np.concatenate([box_position, box_quat])

    # target box
    x_range = [-0.01, -0.01]
    y_range = [0.6, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    target_box_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    target_box_quat = np.array([1, 0, 0, 0])
    target_box_pose = np.concatenate([target_box_position, target_box_quat])


    drawer_initial_pose= [0.0] 

    return box_pose, target_box_pose, drawer_initial_pose


def sample_stack_pose():
    """
    Sample random poses for the three blocks in cupboard style
    """
    # Green block (base block)
    x_range = [-0.05, 0.05]
    y_range = [0.42, 0.48]  
    z_range = [0.025, 0.025]
    
    ranges = np.vstack([x_range, y_range, z_range])
    green_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    
    green_quat = np.array([1, 0, 0, 0])
    green_pose = np.concatenate([green_position, green_quat])
    
    # Red block
    x_range = [-0.16, -0.10]
    y_range = [0.42, 0.48]
    z_range = [0.02, 0.02]
    
    ranges = np.vstack([x_range, y_range, z_range])
    red_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    
    red_quat = np.array([1, 0, 0, 0])
    red_pose = np.concatenate([red_position, red_quat])
    
    # Blue block  
    x_range = [0.10, 0.20]
    y_range = [0.42, 0.48]
    z_range = [0.02, 0.02]
    
    ranges = np.vstack([x_range, y_range, z_range])
    blue_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    
    blue_quat = np.array([1, 0, 0, 0])
    blue_pose = np.concatenate([blue_position, blue_quat])
    
    return green_pose, red_pose, blue_pose

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d