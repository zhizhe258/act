#!/usr/bin/env python3
"""
Z Latent Variable Analysis for ACT Models
Analyze Z latent space diversity across different tasks:

actnew project tasks:
- converted_bimanual_aloha_slot_insertion
- converted_bimanual_aloha_peg_insertion
- converted_bimanual_aloha_hook_package

act project tasks (optional, enabled by default):
- sim_transfer_cube_scripted
- sim_insertion_scripted
- sim_cupboard_scripted
- sim_stack_scripted

Usage:
    # Analyze all tasks (default)
    python z_analysis.py
    
    # Analyze only actnew tasks
    python z_analysis.py --actnew_only
"""

import torch
import numpy as np
import os
import pickle
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import sys

# Add detr path
sys.path.append('detr')

from detr.models.detr_vae import build
from einops import rearrange
import torchvision.transforms as transforms

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')


def get_model_args(camera_names=['overhead_cam']):
    """Create model arguments"""
    class Args:
        def __init__(self):
            self.lr = 1e-5
            self.lr_backbone = 1e-5
            self.backbone = 'resnet18'
            self.dilation = False
            self.position_embedding = 'sine'
            self.hidden_dim = 512
            self.dropout = 0.1
            self.nheads = 8
            self.dim_feedforward = 3200
            self.enc_layers = 4
            self.dec_layers = 7
            self.pre_norm = False
            self.num_queries = 100
            self.camera_names = camera_names
            self.masks = False
    
    return Args()


def load_model_direct(ckpt_path, stats_path, camera_names=['overhead_cam']):
    """Directly load DETR model - supports multiple checkpoint formats"""
    # Build model
    args = get_model_args(camera_names)
    model = build(args)
    
    # Load checkpoint - robust checkpoint format handling
    ckpt = torch.load(ckpt_path, map_location='cuda')
    
    # Extract actual state_dict
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        state = ckpt['model']
    else:
        state = ckpt  # Direct state_dict
    
    # Remove possible prefixes
    new_state = {}
    for k, v in state.items():
        new_state[k.replace('model.', '').replace('module.', '')] = v
    
    model.load_state_dict(new_state, strict=False)
    model.cuda()
    model.eval()
    
    # Load statistics
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    return model, stats


def process_image(image_array):
    """Process image data - simplified and consistent preprocessing pipeline"""
    image = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0  # (C,H,W)
    image = image.unsqueeze(0).cuda()  # (1,C,H,W)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image = normalize(image)  # (1,C,H,W)
    # If model needs (B, T, C, H, W), add time dimension here:
    image = image.unsqueeze(1)  # (1,1,C,H,W)
    return image


def extract_z_vectors(model, dataset_dir, stats, camera_name='overhead_cam', num_episodes=50, samples_per_episode=20):
    """Extract Z vectors from dataset, simulating training-time sampling"""
    z_means = []
    z_logvars = []
    
    print(f"Extracting Z vectors from {dataset_dir}...")
    print(f"Processing {num_episodes} episodes, sampling {samples_per_episode} times per episode")
    print(f"Using camera: {camera_name}")
    
    with torch.no_grad():
        for ep_idx in tqdm(range(num_episodes), desc="Extracting Z vectors"):
            ep_path = os.path.join(dataset_dir, f'episode_{ep_idx}.hdf5')
            
            if not os.path.exists(ep_path):
                continue
                
            try:
                # Load episode data
                with h5py.File(ep_path, 'r') as root:
                    episode_len = root['/action'].shape[0]
                    all_qpos = root['/observations/qpos'][:]
                    all_images = root[f'/observations/images/{camera_name}'][:]
                    all_actions = root['/action'][:]
                
                # Multiple sampling per episode (simulating training-time random sampling)
                for sample_idx in range(samples_per_episode):
                    # Randomly select starting timestep (simulating training-time sampling)
                    start_ts = np.random.choice(episode_len)
                    
                    # Get observations at this timestep
                    qpos = all_qpos[start_ts]
                    image = all_images[start_ts]
                    
                    # Get action sequence starting from start_ts
                    actions = all_actions[start_ts:]
                    action_len = episode_len - start_ts
                    
                    # Numerical stability handling
                    eps = 1e-8
                    qpos_mean = torch.from_numpy(stats['qpos_mean']).float().cuda()
                    qpos_std = torch.from_numpy(stats['qpos_std']).float().cuda()
                    action_mean = torch.from_numpy(stats['action_mean']).float().cuda()
                    action_std = torch.from_numpy(stats['action_std']).float().cuda()
                    
                    # Preprocess qpos
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    qpos = (qpos - qpos_mean) / (qpos_std + eps)
                    
                    # Preprocess image
                    image_tensor = process_image(image)
                    
                    # Dynamically infer action dimension
                    action_dim = int(action_mean.numel())
                    
                    # Preprocess actions - pad to original episode length
                    padded_action = np.zeros((episode_len, action_dim), dtype=np.float32)
                    padded_action[:action_len] = actions
                    
                    # Create padding mask
                    is_pad = np.zeros(episode_len)
                    is_pad[action_len:] = 1
                    
                    # Convert to tensor and normalize
                    actions_tensor = torch.from_numpy(padded_action).float().cuda().unsqueeze(0)
                    actions_tensor = (actions_tensor - action_mean) / (action_std + eps)
                    is_pad_tensor = torch.from_numpy(is_pad).bool().cuda().unsqueeze(0)
                    
                    # Truncate to model's num_queries length
                    target_len = 100  # num_queries
                    if episode_len > target_len:
                        actions_tensor = actions_tensor[:, :target_len, :]
                        is_pad_tensor = is_pad_tensor[:, :target_len]
                    else:
                        # If episode length is less than num_queries, pad
                        pad_len = target_len - episode_len
                        actions_tensor = torch.cat([actions_tensor, torch.zeros(1, pad_len, action_dim).cuda()], dim=1)
                        is_pad_tensor = torch.cat([is_pad_tensor, torch.ones(1, pad_len, dtype=torch.bool).cuda()], dim=1)
                    
                    # Forward pass to get Z
                    _, _, (mu, logvar) = model(qpos, image_tensor, None, actions_tensor, is_pad_tensor)
                    
                    if mu is not None and logvar is not None:
                        # Confirm Z dimension is 32 (and compatible shapes)
                        if mu.dim() == 3:  # (B, T, Z) -> only aggregate valid steps
                            valid = (~is_pad_tensor).float()  # (B, T)
                            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                            mu = (mu * valid.unsqueeze(-1)).sum(dim=1) / denom
                            if logvar.dim() == 3:
                                logvar = (logvar * valid.unsqueeze(-1)).sum(dim=1) / denom
                        
                        assert mu.dim() == 2, f"Unexpected mu shape: {mu.shape}"
                        z_dim = mu.shape[-1]
                        if z_dim != 32:
                            print(f"[Warning] Current model z_dim={z_dim}, not 32.")
                        
                        z_means.append(mu.cpu().numpy())
                        z_logvars.append(logvar.cpu().numpy())
                        
            except Exception as e:
                print(f"Error processing episode {ep_idx}: {e}")
                continue
    
    if len(z_means) > 0:
        z_means = np.vstack(z_means)
        z_logvars = np.vstack(z_logvars)
        print(f"Successfully extracted {len(z_means)} Z vectors")
        return z_means, z_logvars
    else:
        return None, None


class TaskZAnalyzer:
    """Z Latent Variable Analyzer for Multiple Tasks"""
    
    def __init__(self, output_dir='z_analysis_results', include_act_tasks=True):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # actnew project base path
        actnew_base = '/home/zzt/act1/actnew'
        # act project base path  
        act_base = '/home/zzt/act'
        
        # Task configurations - actnew tasks
        self.tasks = {
            'slot_insertion': {
                'ckpt_path': f'{actnew_base}/checkpoints/converted_bimanual_aloha_slot_insertion/policy_best.ckpt',
                'stats_path': f'{actnew_base}/checkpoints/converted_bimanual_aloha_slot_insertion/dataset_stats.pkl',
                'dataset_dir': f'{actnew_base}/data/converted_bimanual_aloha_slot_insertion',
                'camera_name': 'overhead_cam',
                'label': 'Slot Insertion',
                'color': 'blue'
            },
            'peg_insertion': {
                'ckpt_path': f'{actnew_base}/checkpoints/converted_bimanual_aloha_peg_insertion/policy_best.ckpt',
                'stats_path': f'{actnew_base}/checkpoints/converted_bimanual_aloha_peg_insertion/dataset_stats.pkl',
                'dataset_dir': f'{actnew_base}/data/converted_bimanual_aloha_peg_insertion',
                'camera_name': 'overhead_cam',
                'label': 'Peg Insertion',
                'color': 'red'
            },
            'hook_package': {
                'ckpt_path': f'{actnew_base}/checkpoints/converted_bimanual_aloha_hook_package/policy_best.ckpt',
                'stats_path': f'{actnew_base}/checkpoints/converted_bimanual_aloha_hook_package/dataset_stats.pkl',
                'dataset_dir': f'{actnew_base}/data/converted_bimanual_aloha_hook_package',
                'camera_name': 'overhead_cam',
                'label': 'Hook Package',
                'color': 'green'
            }
        }
        
        # Add act project tasks if requested
        if include_act_tasks:
            act_tasks = {
                'sim_transfer_cube': {
                    'ckpt_path': f'{act_base}/checkpoints/sim_transfer_cube_scripted/model/policy_best.ckpt',
                    'stats_path': f'{act_base}/checkpoints/sim_transfer_cube_scripted/model/dataset_stats.pkl',
                    'dataset_dir': f'{act_base}/data/sim_transfer_cube_scripted',
                    'camera_name': 'top',
                    'label': 'Sim Transfer Cube',
                    'color': 'purple'
                },
                'sim_insertion': {
                    'ckpt_path': f'{act_base}/checkpoints/sim_insertion_random/model/policy_best.ckpt',
                    'stats_path': f'{act_base}/checkpoints/sim_insertion_random/model/dataset_stats.pkl',
                    'dataset_dir': f'{act_base}/data/sim_insertion_random',
                    'camera_name': 'top',
                    'label': 'Sim Insertion',
                    'color': 'orange'
                },
                'sim_cupboard': {
                    'ckpt_path': f'{act_base}/checkpoints/sim_cupboard_scripted/model/policy_best.ckpt',
                    'stats_path': f'{act_base}/checkpoints/sim_cupboard_scripted/model/dataset_stats.pkl',
                    'dataset_dir': f'{act_base}/data/sim_cupboard_scripted',
                    'camera_name': 'top',
                    'label': 'Sim Cupboard',
                    'color': 'cyan'
                },
                'sim_stack': {
                    'ckpt_path': f'{act_base}/checkpoints/sim_stack_scripted/model/policy_best.ckpt',
                    'stats_path': f'{act_base}/checkpoints/sim_stack_scripted/model/dataset_stats.pkl',
                    'dataset_dir': f'{act_base}/data/sim_stack_scripted',
                    'camera_name': 'top',
                    'label': 'Sim Stack',
                    'color': 'brown'
                }
            }
            self.tasks.update(act_tasks)
        
        self.z_results = {}
    
    def run_analysis(self, num_episodes=50, samples_per_episode=20):
        """Run complete analysis"""
        print("=" * 60)
        print("ACT Model Z Latent Variable Analysis")
        print(f"Analyzing {len(self.tasks)} tasks from actnew and act projects")
        print("=" * 60)
        
        # Extract Z vectors for each task
        for task_name, config in self.tasks.items():
            print(f"\n{'='*60}")
            print(f"Processing {config['label']}...")
            print(f"{'='*60}")
            
            if not os.path.exists(config['ckpt_path']):
                print(f"Checkpoint not found: {config['ckpt_path']}, skipping...")
                continue
            if not os.path.exists(config['stats_path']):
                print(f"Stats not found: {config['stats_path']}, skipping...")
                continue
            if not os.path.exists(config['dataset_dir']):
                print(f"Dataset not found: {config['dataset_dir']}, skipping...")
                continue
            
            try:
                # Load model
                model, stats = load_model_direct(
                    config['ckpt_path'], 
                    config['stats_path'],
                    camera_names=[config['camera_name']]
                )
                print(f"Successfully loaded {config['label']} model")
                
                # Extract Z vectors
                z_means, z_logvars = extract_z_vectors(
                    model, 
                    config['dataset_dir'], 
                    stats,
                    camera_name=config['camera_name'],
                    num_episodes=num_episodes, 
                    samples_per_episode=samples_per_episode
                )
                
                if z_means is not None:
                    self.z_results[task_name] = {
                        'z_means': z_means,
                        'z_logvars': z_logvars,
                        'label': config['label'],
                        'color': config['color']
                    }
                
                # Clean up memory
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {config['label']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(self.z_results) == 0:
            print("Error: Failed to extract any Z vectors")
            return
        
        # Visualization and analysis
        self.visualize_pca()
        self.visualize_tsne()
        per_dim_results = self.quantitative_analysis()
        
        # Create per-dimension variance visualization
        if per_dim_results:
            self.create_per_dimension_visualization(per_dim_results)
        
        print(f"\n{'='*60}")
        print("Analysis completed!")
        print(f"Results saved to: {self.output_dir}/")
        print(f"{'='*60}")
    
    def visualize_pca(self):
        """PCA visualization results"""
        print("\nCreating PCA visualization chart...")
        
        # ========== 1. Calculate per-task PCA explained variance ==========
        per_task_pca_results = {}
        print("\nCalculating per-task PCA explained variance...")
        
        for task_name, data in self.z_results.items():
            z_means = data['z_means']
            label = data['label']
            color = data['color']
            
            # Standardize each task's data independently
            scaler = StandardScaler()
            z_scaled = scaler.fit_transform(z_means)
            
            # PCA for this task
            pca_task = PCA(n_components=min(32, z_means.shape[0]))
            pca_task.fit(z_scaled)
            
            # Calculate metrics
            explained_var_2d = pca_task.explained_variance_ratio_[:2].sum()
            explained_var_cumsum = np.cumsum(pca_task.explained_variance_ratio_)
            n_components_90 = np.searchsorted(explained_var_cumsum, 0.9) + 1
            
            per_task_pca_results[task_name] = {
                'label': label,
                'color': color,
                'explained_variance_ratio': pca_task.explained_variance_ratio_.tolist(),
                'explained_var_2d': float(explained_var_2d),
                'total_variance': float(np.var(z_means)),
                'n_components_90': int(n_components_90),
                'pc1_ratio': float(pca_task.explained_variance_ratio_[0]),
                'pc2_ratio': float(pca_task.explained_variance_ratio_[1])
            }
            
            print(f"  {label}: PC1+PC2 explains {explained_var_2d:.3f}, "
                  f"needs {n_components_90} components for 90%")
        
        # ========== 2. Prepare combined data for joint PCA visualization ==========
        all_z = []
        
        for task_name, data in self.z_results.items():
            all_z.append(data['z_means'])
        
        all_z = np.vstack(all_z)
        
        # Standardize combined data
        scaler = StandardScaler()
        all_z_scaled = scaler.fit_transform(all_z)
        
        # PCA on combined data
        print("Performing PCA dimensionality reduction...")
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(all_z_scaled)
        
        # ========== 3. Create simple PCA scatter plot ==========
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        start_idx = 0
        for task_name, data in self.z_results.items():
            end_idx = start_idx + len(data['z_means'])
            ax.scatter(z_pca[start_idx:end_idx, 0], z_pca[start_idx:end_idx, 1],
                       c=data['color'], label=data['label'], alpha=0.6, s=30)
            start_idx = end_idx
        
        ax.set_title('Z Latent Variable PCA Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'z_analysis_pca.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA visualization saved to {save_path}")
        plt.close()
        
        # ========== 4. Save per-task PCA results to JSON ==========
        # Add combined PCA info
        pca_results = {
            'combined': {
                'pc1_ratio': float(pca.explained_variance_ratio_[0]),
                'pc2_ratio': float(pca.explained_variance_ratio_[1]),
                'explained_var_2d': float(pca.explained_variance_ratio_.sum())
            },
            'per_task': per_task_pca_results
        }
        
        pca_json_path = os.path.join(self.output_dir, 'z_pca_per_task.json')
        with open(pca_json_path, 'w', encoding='utf-8') as f:
            json.dump(pca_results, f, ensure_ascii=False, indent=2)
        print(f"Per-task PCA results saved to {pca_json_path}")
    
    def visualize_tsne(self):
        """t-SNE visualization results"""
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("t-SNE not available, skipping...")
            return
        
        print("\nCreating t-SNE visualization charts...")
        
        # Prepare data
        all_z = []
        
        for task_name, data in self.z_results.items():
            all_z.append(data['z_means'])
        
        all_z = np.vstack(all_z)
        
        # Standardize
        scaler = StandardScaler()
        all_z_scaled = scaler.fit_transform(all_z)
        
        # t-SNE dimensionality reduction
        print("Performing t-SNE dimensionality reduction (this may take a while)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        z_tsne = tsne.fit_transform(all_z_scaled)
        
        # Create chart
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        start_idx = 0
        for task_name, data in self.z_results.items():
            end_idx = start_idx + len(data['z_means'])
            ax.scatter(z_tsne[start_idx:end_idx, 0], z_tsne[start_idx:end_idx, 1],
                       c=data['color'], label=data['label'], alpha=0.6, s=30)
            start_idx = end_idx
        
        ax.set_title('Z Latent Variable t-SNE Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'z_analysis_tsne.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to {save_path}")
        plt.close()
    
    def quantitative_analysis(self):
        """Quantitative analysis"""
        print("\n" + "=" * 60)
        print("Z Latent Variable Quantitative Analysis Results")
        print("=" * 60)
        
        results = {}
        per_dim_results = {}
        
        for task_name, data in self.z_results.items():
            z_means = data['z_means']
            
            # Calculate variance for each dimension
            var_vec = np.var(z_means, axis=0)  # shape: (32,)
            cov_matrix = np.cov(z_means.T)
            eigenvals, _ = np.linalg.eigh(cov_matrix)
            eigenvals = np.sort(eigenvals)[::-1]
            
            results[task_name] = {
                'label': data['label'],
                'num_samples': len(z_means),
                'z_dim': z_means.shape[1],
                'mean_variance': float(np.mean(var_vec)),
                'total_variance': float(np.sum(var_vec)),
                'max_variance': float(np.max(var_vec)),
                'min_variance': float(np.min(var_vec)),
                'effective_dimensions': int(np.sum(eigenvals > 0.01)),
                'top_5_eigenvalues': eigenvals[:5].tolist()
            }
            
            # Save detailed variance analysis for each dimension
            top_dims_indices = np.argsort(var_vec)[-5:][::-1]  # Top 5 most variable dimensions
            per_dim_results[task_name] = {
                'label': data['label'],
                'per_dimension_variances': var_vec.tolist(),
                'top_5_variable_dims': {
                    'indices': top_dims_indices.tolist(),
                    'variances': var_vec[top_dims_indices].tolist()
                },
                'variance_statistics': {
                    'min': float(np.min(var_vec)),
                    'max': float(np.max(var_vec)),
                    'mean': float(np.mean(var_vec)),
                    'std': float(np.std(var_vec)),
                    'num_active_dims': int(np.sum(var_vec > 1e-6))
                }
            }
        
        # Print results
        print(f"\n{'Task':<20} {'Samples':<10} {'Z Dim':<8} {'Total Var':<12} {'Mean Var':<12} {'Eff Dims':<10}")
        print("-" * 80)
        
        for task_name, result in results.items():
            print(f"{result['label']:<20} {result['num_samples']:<10} "
                  f"{result['z_dim']:<8} {result['total_variance']:<12.4f} "
                  f"{result['mean_variance']:<12.6f} {result['effective_dimensions']:<10}")
        
        # Compare tasks
        print(f"\n{'='*60}")
        print("Task Comparison Analysis")
        print(f"{'='*60}")
        
        task_names = list(results.keys())
        for i, task1 in enumerate(task_names):
            for task2 in task_names[i+1:]:
                var1 = results[task1]['total_variance']
                var2 = results[task2]['total_variance']
                ratio = var1 / var2 if var2 > 0 else float('inf')
                
                print(f"\n{results[task1]['label']} vs {results[task2]['label']}:")
                print(f"  Total Variance: {var1:.4f} vs {var2:.4f} (ratio: {ratio:.2f})")
                print(f"  Effective Dims: {results[task1]['effective_dimensions']} vs {results[task2]['effective_dimensions']}")
        
        # Save results
        save_path = os.path.join(self.output_dir, 'z_analysis_results.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {save_path}")
        
        # Save per-dimension detailed analysis results
        per_dim_path = os.path.join(self.output_dir, 'z_per_dimension_analysis.json')
        with open(per_dim_path, 'w', encoding='utf-8') as f:
            json.dump(per_dim_results, f, ensure_ascii=False, indent=2)
        print(f"Per-dimension analysis saved to {per_dim_path}")
        
        # Print per-dimension analysis summary
        self.print_per_dimension_summary(per_dim_results)
        
        return per_dim_results
    
    def print_per_dimension_summary(self, per_dim_results):
        """Print per-dimension variance analysis summary"""
        print("\n" + "=" * 60)
        print("Per-Dimension Variance Analysis")
        print("=" * 60)
        
        for task_name, data in per_dim_results.items():
            print(f"\n【{data['label']}】:")
            stats = data['variance_statistics']
            print(f"   Variance range: {stats['min']:.6f} - {stats['max']:.6f}")
            print(f"   Mean variance: {stats['mean']:.6f}")
            print(f"   Variance std: {stats['std']:.6f}")
            print(f"   Active dimensions: {stats['num_active_dims']}/32")
            
            print(f"   Top 5 most variable dimensions:")
            for i, (dim_idx, variance) in enumerate(zip(data['top_5_variable_dims']['indices'],
                                                        data['top_5_variable_dims']['variances'])):
                print(f"     #{i + 1}: Dim {dim_idx:2d} = {variance:.6f}")
    
    def create_per_dimension_visualization(self, per_dim_results):
        """Create visualization charts for per-dimension variance"""
        print("\nCreating per-dimension variance analysis charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Variance bar chart for all dimensions of each task
        ax1 = axes[0, 0]
        x_dims = np.arange(32)
        width = 0.25
        
        for i, (task_name, data) in enumerate(per_dim_results.items()):
            variances = np.array(data['per_dimension_variances'])
            color = self.z_results[task_name]['color']
            ax1.bar(x_dims + i * width, variances, width,
                    label=data['label'], color=color, alpha=0.7)
        
        ax1.set_xlabel('Z Dimension', fontsize=12)
        ax1.set_ylabel('Variance', fontsize=12)
        ax1.set_title('Per-Dimension Variance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_dims + width)
        ax1.set_xticklabels([f'{i}' for i in range(32)], fontsize=8)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Variance distribution histogram
        ax2 = axes[0, 1]
        for task_name, data in per_dim_results.items():
            variances = np.array(data['per_dimension_variances'])
            color = self.z_results[task_name]['color']
            ax2.hist(variances, bins=20, alpha=0.5, color=color,
                     label=data['label'], density=True)
        
        ax2.set_xlabel('Variance Value', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Distribution of Per-Dimension Variances', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative variance plot
        ax3 = axes[1, 0]
        for task_name, data in per_dim_results.items():
            variances = np.array(data['per_dimension_variances'])
            sorted_var = np.sort(variances)[::-1]
            cumsum = np.cumsum(sorted_var) / np.sum(sorted_var)
            color = self.z_results[task_name]['color']
            ax3.plot(range(1, len(cumsum) + 1), cumsum, '-o',
                     color=color, label=data['label'], alpha=0.7, markersize=4)
        
        ax3.set_xlabel('Number of Dimensions (sorted by variance)', fontsize=12)
        ax3.set_ylabel('Cumulative Variance Ratio', fontsize=12)
        ax3.set_title('Cumulative Variance by Dimension', fontsize=14, fontweight='bold')
        ax3.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Box plot comparison
        ax4 = axes[1, 1]
        box_data = []
        labels = []
        colors = []
        for task_name, data in per_dim_results.items():
            box_data.append(data['per_dimension_variances'])
            labels.append(data['label'])
            colors.append(self.z_results[task_name]['color'])
        
        bp = ax4.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Variance', fontsize=12)
        ax4.set_title('Variance Distribution Box Plot', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'z_per_dimension_variance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-dimension variance charts saved to {save_path}")
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Z Latent Variable Analysis for ACT Models')
    parser.add_argument('--num_episodes', type=int, default=50, help='Number of episodes to process')
    parser.add_argument('--samples_per_episode', type=int, default=50, help='Samples per episode')
    parser.add_argument('--output_dir', type=str, default='z_analysis_results', help='Output directory')
    parser.add_argument('--include_act_tasks', action='store_true', default=True,
                        help='Include tasks from /home/zzt/act/act/checkpoints (default: True)')
    parser.add_argument('--actnew_only', action='store_true', default=False,
                        help='Only analyze actnew tasks (exclude act project tasks)')
    args = parser.parse_args()
    
    # Determine whether to include act tasks
    include_act = args.include_act_tasks and not args.actnew_only
    
    analyzer = TaskZAnalyzer(output_dir=args.output_dir, include_act_tasks=include_act)
    analyzer.run_analysis(
        num_episodes=args.num_episodes,
        samples_per_episode=args.samples_per_episode
    )


if __name__ == '__main__':
    main()
