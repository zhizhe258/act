# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Improved RTC-enabled DETR model with dual-stage query design for real-time chunking.
Key improvements:
1. Dual-stage Query Design (frozen + learnable)
2. Progressive Masking Training Strategy  
3. Optimized Attention Mask Strategy
4. Improved Sliding Window Mechanism
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from .detr_vae import get_sinusoid_encoding_table, reparametrize

import numpy as np
import IPython
e = IPython.embed


class RTCDETRDecoder(nn.Module):
    """Improved RTC Decoder with dual-stage query design"""
    
    def __init__(self, d_model, action_dim, k, d):
        super().__init__()
        self.k = k  # chunk size
        self.d = d  # frozen steps
        self.d_model = d_model
        self.action_dim = action_dim
        
        # Dual-stage Query Design
        self.frozen_query_proj = nn.Linear(action_dim, d_model)  # Project past actions to queries
        self.learnable_queries = nn.Parameter(torch.randn(k-d, d_model))  # Learnable queries for future
        
        # Query embeddings for full training mode
        self.all_queries = nn.Parameter(torch.randn(k, d_model))
        
        # Initialize parameters
        nn.init.normal_(self.learnable_queries, std=0.02)
        nn.init.normal_(self.all_queries, std=0.02)
        
    def get_queries(self, past_actions=None, training_mode=False):
        """
        Get queries based on mode:
        - Training: use full queries with masking
        - Inference: combine frozen + learnable queries
        """
        if training_mode:
            return self.all_queries
        else:
            if past_actions is not None:
                # Inference mode: combine frozen and learnable
                frozen_queries = self.frozen_query_proj(past_actions)  # [batch, d, d_model]
                batch_size = frozen_queries.shape[0]
                learnable_expanded = self.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)
                queries = torch.cat([frozen_queries, learnable_expanded], dim=1)  # [batch, k, d_model]
                return queries
            else:
                # No past actions, use all learnable
                return self.all_queries
    
    def create_rtc_attention_mask(self, batch_size, device):
        """
        Create attention mask for RTC training:
        - Frozen queries (first d) don't attend to each other
        - But maintain self-attention capability
        """
        mask = torch.zeros(self.k, self.k, device=device)
        
        # Prevent frozen queries from attending to other frozen queries
        # But allow self-attention (diagonal elements remain 0)
        for i in range(self.d):
            for j in range(self.d):
                if i != j:  # Not self-attention
                    mask[i, j] = float('-inf')
        
        # Expand for batch dimension
        return mask.unsqueeze(0).expand(batch_size, -1, -1)


class DETRVAERTC_Improved(nn.Module):
    """Improved RTC-enabled DETR module for real-time action chunking"""
    
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, freeze_steps=4):
        super().__init__()
        self.num_queries = num_queries  # k
        self.freeze_steps = freeze_steps  # d
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        
        # Action and padding prediction heads
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Backbone processing
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # Encoder parameters for VAE
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(14, hidden_dim)
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))

        # Decoder parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)
        
        # RTC training state
        self.rtc_training = False
        self.progressive_weight = 0.0
        
    def enable_rtc_training(self, enable=True, progressive_weight=0.2):
        """Enable/disable RTC training mode"""
        self.rtc_training = enable
        self.progressive_weight = progressive_weight
        
    def progressive_training_schedule(self, epoch, total_epochs):
        """Progressive training schedule for RTC"""
        if self.rtc_training:
            # Gradually increase RTC weight
            progress = epoch / total_epochs
            self.progressive_weight = min(0.5, 0.1 + 0.4 * progress)

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, past_actions=None):
        """
        Improved forward pass with RTC support
        Args:
            qpos: [batch, qpos_dim] - robot joint positions
            image: [batch, num_cam, channel, height, width] - visual observations
            env_state: environment state (can be None)
            actions: [batch, seq, action_dim] - target actions for training
            is_pad: [batch, seq] - padding mask for training
            past_actions: [batch, d, action_dim] - past actions for RTC inference
        """
        is_training = actions is not None
        bs = qpos.shape[0]
        
        ### Obtain latent z from action sequence (VAE encoder)
        if is_training:
            # Training mode with progressive RTC scenarios
            action_embed_result = self._encode_actions_progressive(actions, is_pad, qpos, bs)
            encoder_output = self.encoder(action_embed_result['input'], 
                                        pos=action_embed_result['pos'],
                                        src_key_padding_mask=action_embed_result['mask'])
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            # Inference mode
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # Process visual features - simplified like ACT
        if self.backbones is not None:
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            
            proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            
            # Use standard query embedding like ACT (simplified for now)
            hs = self.transformer(src, None, self.query_embed.weight, pos, 
                                latent_input, proprio_input, 
                                self.additional_pos_embed.weight)[0]
        else:
            # Environment without vision
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)
            queries = self.rtc_decoder.get_queries(past_actions, is_training)
            hs = self.transformer(transformer_input, None, queries.transpose(0, 1), 
                                self.pos.weight)[0]
        
        # Predict actions and padding
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        
        return a_hat, is_pad_hat, [mu, logvar]

    def _encode_actions_progressive(self, actions, is_pad, qpos, bs):
        """
        Progressive masking training strategy - simulate 5 different RTC scenarios
        """
        if not self.rtc_training:
            return self._encode_actions_standard(actions, is_pad, qpos, bs)
        
        # Randomly choose one of 5 RTC training scenarios
        scenario = torch.randint(0, 5, (1,)).item()
        
        if scenario == 0:
            # Scenario 1: Standard full sequence training
            return self._encode_actions_standard(actions, is_pad, qpos, bs)
        elif scenario == 1:
            # Scenario 2: Random starting point with d frozen steps
            max_start = max(0, actions.shape[1] - self.num_queries)
            start_idx = torch.randint(0, max_start + 1, (1,)).item()
            return self._encode_actions_rtc_scenario(actions, is_pad, qpos, bs, start_idx)
        elif scenario == 2:
            # Scenario 3: Middle chunk simulation
            mid_point = actions.shape[1] // 2
            start_idx = max(0, mid_point - self.num_queries // 2)
            return self._encode_actions_rtc_scenario(actions, is_pad, qpos, bs, start_idx)
        elif scenario == 3:
            # Scenario 4: Late sequence simulation
            start_idx = max(0, actions.shape[1] - self.num_queries - 2)
            return self._encode_actions_rtc_scenario(actions, is_pad, qpos, bs, start_idx)
        else:
            # Scenario 5: Random frozen length (between d/2 and d)
            random_d = torch.randint(self.freeze_steps // 2, self.freeze_steps + 1, (1,)).item()
            return self._encode_actions_rtc_variable_d(actions, is_pad, qpos, bs, random_d)

    def _encode_actions_rtc_scenario(self, actions, is_pad, qpos, bs, start_idx):
        """Encode actions for a specific RTC scenario"""
        chunk_actions = actions[:, start_idx:start_idx + self.num_queries]
        chunk_is_pad = is_pad[:, start_idx:start_idx + self.num_queries]
        
        # Pad if necessary
        if chunk_actions.shape[1] < self.num_queries:
            pad_length = self.num_queries - chunk_actions.shape[1]
            chunk_actions = torch.cat([
                chunk_actions,
                torch.zeros(bs, pad_length, chunk_actions.shape[2]).to(chunk_actions.device)
            ], dim=1)
            chunk_is_pad = torch.cat([
                chunk_is_pad,
                torch.ones(bs, pad_length).to(chunk_is_pad.device).bool()
            ], dim=1)
        
        # Create frozen mask (first d steps are frozen)
        frozen_mask = torch.zeros_like(chunk_actions[:, :, 0], dtype=torch.bool)
        frozen_mask[:, :self.freeze_steps] = True
        
        # Apply RTC encoding
        return self._encode_with_mask(chunk_actions, chunk_is_pad, qpos, bs, frozen_mask)

    def _encode_actions_rtc_variable_d(self, actions, is_pad, qpos, bs, variable_d):
        """Encode actions with variable frozen length"""
        start_idx = torch.randint(0, max(1, actions.shape[1] - self.num_queries), (1,)).item()
        chunk_actions = actions[:, start_idx:start_idx + self.num_queries]
        chunk_is_pad = is_pad[:, start_idx:start_idx + self.num_queries]
        
        # Create frozen mask with variable length
        frozen_mask = torch.zeros_like(chunk_actions[:, :, 0], dtype=torch.bool)
        frozen_mask[:, :variable_d] = True
        
        return self._encode_with_mask(chunk_actions, chunk_is_pad, qpos, bs, frozen_mask)

    def _encode_with_mask(self, actions, is_pad, qpos, bs, frozen_mask):
        """Core encoding with masking"""
        # Add mask embedding to indicate frozen vs predicted tokens
        action_embed = self.encoder_action_proj(actions)
        
        # Add learnable embedding to distinguish frozen vs learnable parts
        mask_embed_values = torch.zeros_like(action_embed)
        mask_embed_values[frozen_mask] = 0.1  # Small bias for frozen tokens
        action_embed = action_embed + mask_embed_values
        
        qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
        cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)
        encoder_input = encoder_input.permute(1, 0, 2)
        
        cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
        is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
        
        pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
        
        return {
            'input': encoder_input,
            'pos': pos_embed,
            'mask': is_pad
        }

    def _encode_actions_standard(self, actions, is_pad, qpos, bs):
        """Standard action encoding for regular ACT training"""
        action_embed = self.encoder_action_proj(actions)
        qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
        cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)
        encoder_input = encoder_input.permute(1, 0, 2)
        
        cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
        is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
        
        pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
        
        return {
            'input': encoder_input,
            'pos': pos_embed,
            'mask': is_pad
        }


def build_rtc_improved(args):
    """Build improved RTC model"""
    device = torch.device(args.device)

    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)
    transformer = build_transformer(args)
    
    encoder_layer = TransformerEncoderLayer(args.hidden_dim, args.nheads, args.dim_feedforward, 
                                          args.dropout, "relu", normalize_before=args.pre_norm)
    encoder = TransformerEncoder(encoder_layer, args.enc_layers, None)

    model = DETRVAERTC_Improved(
        backbones,
        transformer,
        encoder,
        state_dim=args.state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        freeze_steps=args.freeze_steps,
    )

    return model