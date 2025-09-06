# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from ..util.misc import NestedTensor, is_main_process

try:
    import torch.hub
    import timm
    DINOV2_AVAILABLE = True
    USE_TIMM = True
except ImportError:
    DINOV2_AVAILABLE = False
    USE_TIMM = False

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class DinoV2Backbone(nn.Module):
    """DinoV2 backbone for feature extraction (frozen, pretrained only)."""
    def __init__(self, name: str, return_interm_layers: bool = False):
        super().__init__()
        if not DINOV2_AVAILABLE:
            raise ImportError("torch.hub is required for DinoV2")
        
        # Load pretrained DinoV2 model (frozen for feature extraction only)
        if USE_TIMM:
            # Try loading from timm first (more stable)
            timm_name_map = {
                'dinov2_vits14': 'vit_small_patch14_dinov2.lvd142m',
                'dinov2_vitb14': 'vit_base_patch14_dinov2.lvd142m', 
                'dinov2_vitl14': 'vit_large_patch14_dinov2.lvd142m',
                'dinov2_vitg14': 'vit_giant_patch14_dinov2.lvd142m'
            }
            
            if name in timm_name_map:
                try:
                    print(f"Loading {name} from timm...")
                    # Set dynamic img_size to handle arbitrary input sizes
                    self.dinov2 = timm.create_model(timm_name_map[name], pretrained=True, 
                                                  dynamic_img_size=True)
                    self.use_forward_features = True
                except Exception as e:
                    print(f"Failed to load from timm: {e}")
                    print("Falling back to torch.hub...")
                    self.dinov2 = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
                    self.use_forward_features = False
            else:
                self.dinov2 = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
                self.use_forward_features = False
        else:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
            self.use_forward_features = False
        self.dinov2.eval()
        
        # Freeze all parameters - we only use it for feature extraction
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        # Get embedding dimension based on model variant
        dinov2_dims = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768, 
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536
        }
        self.embed_dim = dinov2_dims.get(name, 768)  # default to base model
        self.num_channels = self.embed_dim
        self.return_interm_layers = return_interm_layers
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # DinoV2 requires input dimensions to be divisible by patch size (14)
        patch_size = 14
        target_H = ((H + patch_size - 1) // patch_size) * patch_size
        target_W = ((W + patch_size - 1) // patch_size) * patch_size
        
        # Resize if needed
        if H != target_H or W != target_W:
            x_resized = F.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=False)
        else:
            x_resized = x
        
        # Get features from DinoV2 (returns [B, num_patches+1, embed_dim])
        with torch.no_grad():  # Ensure no gradients for frozen backbone
            if self.use_forward_features:
                # timm models use forward_features
                features = self.dinov2.forward_features(x_resized)
            else:
                # torch.hub DinoV2 models have forward_features method
                features = self.dinov2.forward_features(x_resized)
            
        # Remove CLS token, keep only patch tokens
        patch_features = features[:, 1:]  # [B, num_patches, embed_dim]
        
        # Calculate patch grid size based on actual input dimensions
        num_patches = patch_features.shape[1]
        patch_H = target_H // patch_size
        patch_W = target_W // patch_size
        
        # Verify patch count matches
        assert num_patches == patch_H * patch_W, f"Patch count mismatch: {num_patches} vs {patch_H * patch_W}"
        
        # Reshape to spatial feature map
        spatial_features = patch_features.transpose(1, 2).reshape(
            B, self.embed_dim, patch_H, patch_W
        )  # [B, embed_dim, H_patch, W_patch]
        
        # For compatibility with existing code, return dict like ResNet layers
        if self.return_interm_layers:
            # Simulate multi-scale features by using different interpolations
            layer1 = F.interpolate(spatial_features, scale_factor=2, mode='bilinear', align_corners=False)
            layer2 = F.interpolate(spatial_features, scale_factor=1.5, mode='bilinear', align_corners=False)  
            layer3 = F.interpolate(spatial_features, scale_factor=1.2, mode='bilinear', align_corners=False)
            layer4 = spatial_features
            return {"0": layer1, "1": layer2, "2": layer3, "3": layer4}
        else:
            return {"0": spatial_features}


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    
    # Check if using DinoV2 backbone
    if args.backbone.startswith('dinov2_'):
        backbone = DinoV2Backbone(args.backbone, return_interm_layers)
    else:
        # Use ResNet backbone
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
