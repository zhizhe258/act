#!/usr/bin/env python3
"""
Direct Z latent variable extraction script using DETR model
Bypass ACTPolicy parameter parsing issues, directly build and load model
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

def get_model_args():
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
            self.camera_names = ['top']
            self.masks = False
    
    return Args()

def load_model_direct(ckpt_path, stats_path):
    """Directly load DETR model - supports multiple checkpoint formats"""
    # Build model
    args = get_model_args()
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
    image = torch.from_numpy(image_array).permute(2,0,1).float() / 255.0  # (C,H,W)
    image = image.unsqueeze(0).cuda()  # (1,C,H,W)
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                   std=[0.229,0.224,0.225])
    image = normalize(image)           # (1,C,H,W)
    # If model needs (B, T, C, H, W), add time dimension here:
    image = image.unsqueeze(1)         # (1,1,C,H,W)
    return image

def extract_z_vectors(model, dataset_dir, stats, num_episodes=50, samples_per_episode=20):
    """Extract Z vectors from dataset, simulating training-time sampling"""
    z_means = []
    z_logvars = []
    
    print(f"Extracting Z vectors from {dataset_dir}...")
    print(f"Processing {num_episodes} episodes, sampling {samples_per_episode} times per episode")
    
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
                    all_images = root['/observations/images/top'][:]
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
                        if mu.dim() == 3:   # (B, T, Z) -> only aggregate valid steps
                            valid = (~is_pad_tensor).float()           # (B, T)
                            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                            mu = (mu * valid.unsqueeze(-1)).sum(dim=1) / denom
                            if logvar.dim() == 3:
                                logvar = (logvar * valid.unsqueeze(-1)).sum(dim=1) / denom
                        
                        assert mu.dim() == 2, f"Unexpected mu shape: {mu.shape}"
                        z_dim = mu.shape[-1]
                        if z_dim != 32:
                            print(f"[Warning] Current model z_dim={z_dim}, not 32. Process continues but may not match expectations.")
                        
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

class DirectZExtractor:
    """Direct Z latent variable extractor"""
    
    def __init__(self):
        self.models = {
            'random': {
                'ckpt_path': 'checkpoints/insertion_model/model/policy_best.ckpt',
                'stats_path': 'checkpoints/insertion_model/model/dataset_stats.pkl',
                'dataset_dir': '/home/zzt/act/data/sim_insertion_scripted',
                'label': 'High Diversity Data',
                'color': 'blue'
            },
            'edge': {
                'ckpt_path': 'checkpoints/insertion_model/model_edge/policy_best.ckpt',
                'stats_path': 'checkpoints/insertion_model/model_edge/dataset_stats.pkl',
                'dataset_dir': '/home/zzt/act/data/sim_insertion_scripted_edge',
                'label': 'Edge Case Data',
                'color': 'red'
            },
            'similar': {
                'ckpt_path': 'checkpoints/insertion_model/model_similar/policy_best.ckpt',
                'stats_path': 'checkpoints/insertion_model/model_similar/dataset_stats.pkl',
                'dataset_dir': '/home/zzt/act/data/sim_insertion_scripted_similar',
                'label': 'Similar Data',
                'color': 'green'
            }
        }
        
        self.z_results = {}
    
    def run_analysis(self, num_episodes=50, samples_per_episode=20):
        """Run complete analysis"""
        print("Direct DETR Model Z Latent Variable Analysis")
        print("=" * 50)
        
        # Extract Z vectors
        for model_name, config in self.models.items():
            print(f"\nProcessing {config['label']} model...")
            
            if not os.path.exists(config['ckpt_path']) or not os.path.exists(config['stats_path']):
                print(f"File does not exist, skipping {config['label']}")
                continue
                
            if not os.path.exists(config['dataset_dir']):
                print(f"Dataset directory does not exist, skipping {config['label']}")
                continue
            
            try:
                # Load model
                model, stats = load_model_direct(config['ckpt_path'], config['stats_path'])
                print(f"Successfully loaded {config['label']} model")
                
                # Extract Z vectors
                z_means, z_logvars = extract_z_vectors(model, config['dataset_dir'], stats, num_episodes, samples_per_episode)
                
                if z_means is not None:
                    self.z_results[model_name] = {
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
                continue
        
        if len(self.z_results) == 0:
            print("Error: Failed to extract any Z vectors")
            return
        
        # Visualization and analysis
        self.visualize_results()
        per_dim_results = self.quantitative_analysis()
        
        # Create per-dimension variance visualization
        if per_dim_results:
            self.create_per_dimension_visualization(per_dim_results)
        
        print("\nAnalysis completed!")
    
    def visualize_results(self):
        """PCA visualization results"""
        print("\nCreating PCA visualization charts...")
        
        # Prepare data
        all_z = []
        all_labels = []
        
        for model_name, data in self.z_results.items():
            z_means = data['z_means']
            label = data['label']
            
            all_z.append(z_means)
            all_labels.extend([label] * len(z_means))
        
        all_z = np.vstack(all_z)
        
        # Standardize
        scaler = StandardScaler()
        all_z_scaled = scaler.fit_transform(all_z)
        
        # Create single PCA chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # PCA dimensionality reduction
        print("Performing PCA dimensionality reduction...")
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(all_z_scaled)
        
        # Plot PCA scatter plot
        start_idx = 0
        for model_name, data in self.z_results.items():
            end_idx = start_idx + len(data['z_means'])
            ax.scatter(z_pca[start_idx:end_idx, 0], z_pca[start_idx:end_idx, 1],
                      c=data['color'], label=data['label'], alpha=0.7, s=50)
            start_idx = end_idx
        
        ax.set_title(f'Z Latent Variable PCA Analysis\n(Explained Variance: {pca.explained_variance_ratio_.sum():.3f})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('z_analysis_pca.png', dpi=300, bbox_inches='tight')
        print("PCA visualization results saved to z_analysis_pca.png")
        plt.close()
        
    def quantitative_analysis(self):
        """Quantitative analysis"""
        print("\n" + "="*60)
        print("ACT Model Z Latent Analysis Results")
        print("="*60)
        
        results = {}
        per_dim_results = {}
        
        for model_name, data in self.z_results.items():
            z_means = data['z_means']
            
            # Calculate variance for each dimension
            var_vec = np.var(z_means, axis=0)  # shape: (32,)
            cov_matrix = np.cov(z_means.T)
            eigenvals, _ = np.linalg.eigh(cov_matrix)
            eigenvals = np.sort(eigenvals)[::-1]
            
            results[model_name] = {
                'label': data['label'],
                'num_samples': len(z_means),
                'mean_variance': np.mean(var_vec),
                'total_variance': np.sum(var_vec),
                'effective_dimensions': np.sum(eigenvals > 0.01)
            }
            
            # Save detailed variance analysis for each dimension
            top_dims_indices = np.argsort(var_vec)[-5:][::-1]  # Top 5 most variable dimensions
            per_dim_results[model_name] = {
                'label': data['label'],
                'per_dimension_variances': var_vec.tolist(),
                'top_5_variable_dims': {
                    'indices': top_dims_indices.tolist(),
                    'variances': var_vec[top_dims_indices].tolist()
                },
                'variance_statistics': {
                    'min': float(np.min(var_vec)),
                    'max': float(np.max(var_vec)),
                    'std': float(np.std(var_vec)),
                    'num_active_dims': int(np.sum(var_vec > 1e-6))
                }
            }
        
        # Print results
        print(f"{'Model':<20} {'Samples':<8} {'Total Var':<12} {'Mean Var':<12} {'Eff Dims':<10}")
        print("-" * 70)
        
        for model_name, result in results.items():
            print(f"{result['label']:<15} {result['num_samples']:<8} "
                  f"{result['total_variance']:<12.4f} {result['mean_variance']:<12.4f} "
                  f"{result['effective_dimensions']:<10}")
        
        # Verify hypothesis
        if 'random' in results and 'similar' in results:
            random_var = results['random']['total_variance']
            similar_var = results['similar']['total_variance']
            ratio = random_var / similar_var
            
            print(f"\nKey Findings:")
            print(f"- High Diversity Data Total Variance: {random_var:.4f}")
            print(f"- Similar Data Total Variance: {similar_var:.4f}")
            print(f"- Variance Ratio: {ratio:.2f}:1")
            
            if ratio > 1.5:
                print(f"✓ Hypothesis Confirmed: High diversity data produces more diverse latent representations")
                print(f"  This validates the data diversity - model generalization hypothesis")
            else:
                print(f"? Results require further analysis")
        
        # Save overall results
        with open('z_analysis_results.json', 'w', encoding='utf-8') as f:
            serializable_results = {}
            for model_name, result in results.items():
                serializable_results[model_name] = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) 
                                                   for k, v in result.items()}
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # Save per-dimension detailed analysis results  
        with open('z_per_dimension_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(per_dim_results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved: z_analysis_results.json")
        print(f"Per-dimension analysis saved: z_per_dimension_analysis.json")
        
        # Print per-dimension analysis summary
        self.print_per_dimension_summary(per_dim_results)
        
        return per_dim_results

    def print_per_dimension_summary(self, per_dim_results):
        """Print per-dimension variance analysis summary"""
        print("\n" + "="*60)
        print("Per-Dimension Variance Analysis")
        print("="*60)
        
        for model_name, data in per_dim_results.items():
            print(f"\n {data['label']}:")
            stats = data['variance_statistics']
            print(f"   Variance range: {stats['min']:.6f} - {stats['max']:.6f}")
            print(f"   Variance std: {stats['std']:.6f}")
            print(f"   Active dimensions: {stats['num_active_dims']}/32")
            
            print(f"   Top 5 most variable dimensions:")
            for i, (dim_idx, variance) in enumerate(zip(data['top_5_variable_dims']['indices'], 
                                                       data['top_5_variable_dims']['variances'])):
                print(f"     #{i+1}: Dim {dim_idx:2d} = {variance:.6f}")
        
        # Dimension comparison analysis
        if len(per_dim_results) >= 2:
            print(f"\n🔍 Dimension difference comparison:")
            model_names = list(per_dim_results.keys())
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    var1 = np.array(per_dim_results[model1]['per_dimension_variances'])
                    var2 = np.array(per_dim_results[model2]['per_dimension_variances'])
                    
                    # Find dimension with largest difference
                    diff = np.abs(var1 - var2)
                    max_diff_dim = np.argmax(diff)
                    
                    print(f"   {per_dim_results[model1]['label']} vs {per_dim_results[model2]['label']}:")
                    print(f"     Largest difference dimension: Dim {max_diff_dim} (difference: {diff[max_diff_dim]:.6f})")
                    print(f"     Average variance difference: {np.mean(diff):.6f}")

    def create_per_dimension_visualization(self, per_dim_results):
        """Create visualization charts for per-dimension variance"""
        print("\nCreating per-dimension variance analysis charts...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. Variance bar chart for all dimensions of each model
        x_dims = np.arange(32)
        width = 0.25
        
        for i, (model_name, data) in enumerate(per_dim_results.items()):
            variances = np.array(data['per_dimension_variances'])
            color = self.z_results[model_name]['color']
            ax1.bar(x_dims + i*width, variances, width, 
                   label=data['label'], color=color, alpha=0.7)
        
        ax1.set_xlabel('Z Dimension', fontsize=12)
        ax1.set_ylabel('Variance', fontsize=12)
        ax1.set_title('Per-Dimension Variance Comparison Across Models', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_dims + width)
        ax1.set_xticklabels([f'{i}' for i in range(32)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Variance distribution histogram
        for model_name, data in per_dim_results.items():
            variances = np.array(data['per_dimension_variances'])
            color = self.z_results[model_name]['color']
            ax2.hist(variances, bins=20, alpha=0.6, color=color, 
                    label=data['label'], density=True)
        
        ax2.set_xlabel('Variance Value', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Distribution of Per-Dimension Variances', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('z_per_dimension_variance.png', dpi=300, bbox_inches='tight')
        print("Per-dimension variance analysis charts saved to z_per_dimension_variance.png")
        plt.close()

def main():
    extractor = DirectZExtractor()
    # Use all 50 episodes, sample 20 times per episode = total 1000 Z vector samples
    extractor.run_analysis(num_episodes=50, samples_per_episode=20)

if __name__ == '__main__':
    main()
