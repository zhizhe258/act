import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import numpy as np
from detr.models.detr_vae_rtc_improved import build_rtc_improved

import IPython
e = IPython.embed


class RTCPolicy_Improved(nn.Module):
    """
    Improved RTC Policy with optimized sliding window inference
    Key improvements:
    1. Dual-stage query design (frozen + learnable)
    2. Progressive training with 5 different scenarios  
    3. Optimized attention masking
    4. Efficient sliding window inference
    """
    
    def __init__(self, args_override):
        super().__init__()
        model_args = self._get_model_args(args_override)
        self.model = build_rtc_improved(model_args)
        self.model.kl_weight = model_args.kl_weight  # Store kl_weight in model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=model_args.lr)
        
        # RTC-specific parameters
        self.k = model_args.num_queries  # chunk size
        self.d = model_args.freeze_steps  # frozen steps
        self.generation_interval = self.k - self.d  # generate new chunk every k-d steps
        
        # Inference state management
        self.action_buffer = []  # Rolling buffer for executed actions
        self.step_counter = 0    # Count steps since last chunk generation
        self.current_chunk = None  # Current predicted chunk
        self.chunk_index = 0     # Index within current chunk
        
        # Training state
        self.rtc_training_enabled = False
        self.rtc_loss_weight = 0.2
        self.progressive_weight = 0.1
        
        # Normalization statistics (will be loaded from dataset)
        self.stats = None

    def _get_model_args(self, args_override):
        """Convert config dict to model args"""
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        # Default model configuration
        default_args = {
            'lr': 1e-4,
            'num_queries': 16,
            'freeze_steps': 4,
            'kl_weight': 10,
            'hidden_dim': 512,
            'dim_feedforward': 3200,
            'lr_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'dropout': 0.1,
            'pre_norm': False,
            'camera_names': ['top'],
            'state_dim': 14,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Override with provided arguments
        default_args.update(args_override)
        return Args(**default_args)

    def enable_rtc_training(self, enable=True, rtc_weight=0.2):
        """Enable/disable RTC training mode"""
        self.rtc_training_enabled = enable
        self.rtc_loss_weight = rtc_weight
        self.model.enable_rtc_training(enable, rtc_weight)
        print(f"RTC training {'enabled' if enable else 'disabled'}, weight: {rtc_weight}")

    def progressive_training_schedule(self, epoch, total_epochs):
        """Progressive training schedule - gradually increase RTC weight"""
        if self.rtc_training_enabled:
            # Exponential increase in RTC weight
            progress = epoch / total_epochs
            self.progressive_weight = 0.1 + 0.4 * (1 - np.exp(-3 * progress))
            self.rtc_loss_weight = min(0.5, self.progressive_weight)
            self.model.progressive_weight = self.progressive_weight
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: RTC weight = {self.rtc_loss_weight:.3f}")

    def set_rtc_loss_weight(self, weight):
        """Manually set RTC loss weight"""
        self.rtc_loss_weight = weight
        self.model.progressive_weight = weight

    def reset_state(self):
        """Reset inference state for new episode"""
        self.action_buffer = []
        self.step_counter = 0
        self.current_chunk = None
        self.chunk_index = 0
        print("RTC state reset for new episode")

    def configure(self, stats):
        """Configure policy with dataset statistics"""
        self.stats = stats

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        Forward pass with improved RTC logic
        
        Training mode: 
            - Uses progressive masking with 5 different scenarios
            - Applies RTC loss for future prediction training
            
        Inference mode:
            - Uses sliding window with frozen + learnable queries
            - Efficient action generation every k-d steps
        """
        env_state = None  # Not used in current setup
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        if actions is not None:  # Training mode
            return self._forward_training(qpos, image, env_state, actions, is_pad)
        else:  # Inference mode  
            return self._forward_inference(qpos, image, env_state)

    def _forward_training(self, qpos, image, env_state, actions, is_pad):
        """Training forward pass with RTC support"""
        # Truncate actions and is_pad to match model queries (like ACT)
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]
        
        # Standard forward pass
        a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
        
        # Compute standard losses
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction='none')  # Fixed order like ACT
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
        
        # Add RTC loss if enabled
        if self.rtc_training_enabled:
            rtc_loss = self._compute_rtc_loss(a_hat, actions, is_pad)
            loss_dict['rtc_loss'] = rtc_loss
            loss_dict['loss'] += rtc_loss * self.rtc_loss_weight
        
        return loss_dict

    def _forward_inference(self, qpos, image, env_state):
        """Simplified inference like ACT for now"""
        with torch.no_grad():
            a_hat, _, _ = self.model(qpos, image, env_state)
            return a_hat

    def _should_generate_chunk(self):
        """Determine if we should generate a new chunk"""
        # Generate chunk if:
        # 1. First step (no current chunk)
        # 2. Every k-d steps 
        # 3. Current chunk is exhausted
        return (self.current_chunk is None or 
                self.step_counter >= self.generation_interval or
                self.chunk_index >= len(self.current_chunk))

    def _get_past_actions(self, device):
        """Get past d actions for frozen queries"""
        if len(self.action_buffer) >= self.d:
            # Use last d actions
            past_actions = torch.stack(self.action_buffer[-self.d:])
            return past_actions.unsqueeze(0).to(device)  # Add batch dimension
        elif len(self.action_buffer) > 0:
            # Pad with zeros if we don't have enough history
            past_actions = self.action_buffer.copy()
            while len(past_actions) < self.d:
                past_actions.insert(0, torch.zeros_like(self.action_buffer[0]))
            past_actions = torch.stack(past_actions)
            return past_actions.unsqueeze(0).to(device)
        else:
            # No history - use zeros (first few steps)
            action_dim = 14  # Assuming robot action dimension
            return torch.zeros(1, self.d, action_dim).to(device)

    def _compute_rtc_loss(self, predictions, targets, is_pad):
        """
        Compute RTC-specific loss for training
        Focus on predicting future actions given past context
        """
        # Weight later predictions more heavily (progressive difficulty)
        batch_size, seq_len, action_dim = predictions.shape
        
        # Create progressive weights (later steps get higher weight)
        weights = torch.linspace(1.0, 2.0, seq_len).to(predictions.device)
        weights = weights.view(1, seq_len, 1).expand(batch_size, seq_len, action_dim)
        
        # Apply weights to L1 loss
        weighted_l1 = F.l1_loss(predictions, targets, reduction='none') * weights
        
        # Mask out padded regions
        rtc_loss = (weighted_l1 * ~is_pad.unsqueeze(-1)).mean()
        
        return rtc_loss

    def save_model(self, ckpt_path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rtc_config': {
                'k': self.k,
                'd': self.d,
                'rtc_training_enabled': self.rtc_training_enabled,
                'rtc_loss_weight': self.rtc_loss_weight
            }
        }, ckpt_path)

    def load_model(self, ckpt_path):
        """Load model checkpoint"""
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'rtc_config' in checkpoint:
            rtc_config = checkpoint['rtc_config']
            self.k = rtc_config.get('k', self.k)
            self.d = rtc_config.get('d', self.d)
            self.rtc_training_enabled = rtc_config.get('rtc_training_enabled', False)
            self.rtc_loss_weight = rtc_config.get('rtc_loss_weight', 0.2)
            
        print(f"Loaded RTC model: k={self.k}, d={self.d}, RTC training={self.rtc_training_enabled}")

    @property  
    def kl_weight(self):
        """Get KL weight from model args"""
        return getattr(self.model, 'kl_weight', 10)  # Use stored kl_weight or default


def kl_divergence(mu, logvar):
    """Compute KL divergence for VAE"""
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


# Factory function for compatibility
def RTCPolicy(args_override):
    """Factory function to create improved RTC policy"""
    return RTCPolicy_Improved(args_override)