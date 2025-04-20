import copy
from itertools import product
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

from ..layers import (ASPP, DeformableSqueezeAttention, DeformableTransformerLayer,
                      LearnableSqueezePositionalEncoding, TransformerLayer, Upsample, DeformConv3d)
from ..projections import VoxelProposalLayer
from ..utils import (cumprod, flatten_fov_from_voxels, flatten_multi_scale_feats, generate_grid,
                     get_level_start_index, index_fov_back_to_voxels, interpolate_flatten,
                     nchw_to_nlc, nlc_to_nchw, pix2vox)

class DecoderLayer(nn.Module):

    def __init__(self, embed_dims, num_heads=8, num_levels=3, num_points=4, query_update=True):
        super().__init__()
        self.query_image_cross_defrom_attn = DeformableTransformerLayer(
            embed_dims, num_heads, num_levels, num_points)
        self.scene_query_cross_attn = TransformerLayer(embed_dims, num_heads, mlp_ratio=0)
        self.scene_self_deform_attn = DeformableTransformerLayer(
            embed_dims,
            num_heads,
            num_levels=1,
            num_points=num_points * 2,
            attn_layer=DeformableSqueezeAttention)

        self.query_update = query_update
        if query_update:
            self.query_scene_cross_deform_attn = DeformableTransformerLayer(
                embed_dims,
                num_heads,
                num_levels=1,
                num_points=num_points * 2,
                attn_layer=DeformableSqueezeAttention,
                mlp_ratio=0)
            self.query_self_attn = TransformerLayer(embed_dims, num_heads)

    def forward(self,
                scene_embed,
                inst_queries,
                feats,
                scene_pos=None,
                inst_pos=None,
                ref_2d=None,
                ref_3d=None,
                ref_vox=None,
                fov_mask=None):
        scene_embed_fov = flatten_fov_from_voxels(scene_embed, fov_mask)
        scene_pos_fov = flatten_fov_from_voxels(scene_pos,
                                                fov_mask) if scene_pos is not None else None
        scene_embed_flatten, scene_shape = flatten_multi_scale_feats([scene_embed])
        scene_level_index = get_level_start_index(scene_shape)
        feats_flatten, feat_shapes = flatten_multi_scale_feats(feats)
        
        feats_level_index = get_level_start_index(feat_shapes)
        inst_queries = self.query_image_cross_defrom_attn(
            inst_queries,
            feats_flatten,
            query_pos=inst_pos,
            ref_pts=ref_2d,
            spatial_shapes=feat_shapes,
            level_start_index=feats_level_index)

        scene_embed_fov = self.scene_query_cross_attn(scene_embed_fov, inst_queries, inst_queries,
                                                      scene_pos_fov, inst_pos)
        scene_embed_fov = self.scene_self_deform_attn(
            scene_embed_fov,
            scene_embed_flatten,
            query_pos=scene_pos_fov,
            ref_pts=torch.flip(ref_vox[:, fov_mask.squeeze()], dims=[-1]),  # TODO: assert bs == 1
            spatial_shapes=scene_shape,
            level_start_index=scene_level_index)

        scene_embed = index_fov_back_to_voxels(scene_embed, scene_embed_fov, fov_mask)
        scene_embed_flatten, scene_shape = flatten_multi_scale_feats([scene_embed])
        if not self.query_update:
            return scene_embed, inst_queries

        inst_queries = self.query_scene_cross_deform_attn(
            inst_queries,
            scene_embed_flatten,
            query_pos=inst_pos,
            ref_pts=torch.flip(ref_3d, dims=[-1]),
            spatial_shapes=scene_shape,
            level_start_index=scene_level_index)
        inst_queries = self.query_self_attn(inst_queries, query_pos=inst_pos)
        return scene_embed, inst_queries

class MemoryBank(nn.Module):
    def __init__(self, num_tokens=1024, num_keys=3, token_dim=128):
        super(MemoryBank, self).__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_keys = num_keys
        
        # Initialize the memory bank with zeros
        self.register_buffer('keys', torch.zeros(num_tokens, num_keys, token_dim))
        self.register_buffer('tokens', torch.zeros(num_tokens, token_dim))
        self.register_buffer('ages', torch.zeros(num_tokens))
        
    def update_memory_bank(self, new_tokens, ref_pts):
        """
        Update the memory bank with new tokens.
        Assume new_tokens is a tensor of shape [batch_size, token_dim]
        """
        batch_size = new_tokens.size(0)
        
        # Increase the age of existing tokens
        self.ages += 1
        
        # Find the keys for each new token
        dists = torch.cdist(ref_pts, ref_pts, p=2)
        dists.fill_diagonal_(float('inf'))
        closest_points_indices = dists.argsort(dim=1)[:, :self.num_keys]
        new_keys = new_tokens.unsqueeze(1).expand(-1, self.num_keys, -1)
        
        # Concatenate the new tokens and keys with the existing ones
        all_keys = torch.cat((self.keys, new_keys), dim=0)
        all_tokens = torch.cat((self.tokens, new_tokens), dim=0)
        all_ages = torch.cat((self.ages, torch.zeros(batch_size).to(ref_pts.device)), dim=0)
        
        # Normalize the tokens and compute cosine similarity
        norm_tokens = F.normalize(all_tokens, p=2, dim=1)
        cosine_similarity_matrix = torch.mm(norm_tokens, norm_tokens.t())
        
        # Calculate a combined score considering both similarity and age
        diversity_score = -torch.sum(cosine_similarity_matrix, dim=1)
        combined_score = diversity_score + all_ages
        
        # Select the top-k tokens based on the combined score
        _, topk_indices = torch.topk(combined_score, k=self.num_tokens, dim=0, largest=False)
        
        self.keys = all_keys[topk_indices]
        self.tokens = all_tokens[topk_indices]
        self.ages = all_ages[topk_indices]

    def query_tokens(self, incomplete_mask, tokens, ref_pts):
        indices = torch.nonzero(incomplete_mask, as_tuple=False)
        replace_pts = torch.cat((indices.float() / (torch.tensor([128, 128]).to(ref_pts.device) - 1), 
                                    torch.zeros((indices.size(0),1), device=ref_pts.device)), 1)
        dists = torch.cdist(replace_pts, ref_pts, p=2)
        closest_points_indices = dists.argsort(dim=1)[:, :3]
        keys = tokens[closest_points_indices]
        
        keys_flat = keys.view(keys.size(0), -1)  # Shape: [N, 384]
        mem_keys_flat = self.keys.view(1024, -1)  # Shape: [1024, 384]
        similarity = F.cosine_similarity(keys_flat.unsqueeze(1), mem_keys_flat.unsqueeze(0), dim=2)  # Shape: [N, 1024]
        replace_indices = torch.argmax(similarity, dim=1).unsqueeze(1)  # Shape: [N]
        encoding_one_hot = torch.zeros(replace_indices.size(0), self.num_tokens, device=ref_pts.device)
        encoding_one_hot.scatter_(1, replace_indices, 1)
        replace_tokens = torch.matmul(encoding_one_hot, self.tokens)
        return replace_tokens

class MAREDecoder(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_classes,
                 num_layers,
                 num_levels,
                 scene_shape,
                 project_scale,
                 image_shape,
                 voxel_size=0.2,
                 downsample_z=1):
        
        super().__init__()
        self.embed_dims = embed_dims
        scene_shape = [s // project_scale for s in scene_shape]
        if downsample_z != 1:
            self.ori_scene_shape = copy.copy(scene_shape)
            scene_shape[-1] //= downsample_z
        self.scene_shape = scene_shape
        self.num_queries = cumprod(scene_shape)
        self.image_shape = image_shape
        self.voxel_size = voxel_size * project_scale
        self.downsample_z = downsample_z

        self.voxel_proposal = VoxelProposalLayer(embed_dims, scene_shape)
        self.layers = nn.ModuleList([
            DecoderLayer(
                embed_dims,
                num_levels=num_levels,
                query_update=True if i != num_layers - 1 else False) for i in range(num_layers)
        ])

        self.scene_embed = nn.Embedding(self.num_queries, embed_dims)
        self.scene_pos = LearnableSqueezePositionalEncoding((128, 128, 2),
                                                            embed_dims,
                                                            squeeze_dims=(2, 2, 1))
        image_grid = generate_grid(image_shape)
        image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)  # 2(wh), h, w
        self.register_buffer('image_grid', image_grid)
        voxel_grid = generate_grid(scene_shape, normalize=True)
        self.register_buffer('voxel_grid', voxel_grid)

        self.aspp = ASPP(embed_dims, (1, 3, 5))
        # self.aspp = ASPP(embed_dims, (1, 3))
        assert project_scale in (1, 2)
        self.cls_head = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose3d(
                    embed_dims,
                    embed_dims,
                    kernel_size=3,
                    stride=(1, 1, downsample_z),
                    padding=1,
                    output_padding=(0, 0, downsample_z - 1),
                ),
                nn.BatchNorm3d(embed_dims),
                nn.ReLU(),
            ) if downsample_z != 1 else nn.Identity(),
            Upsample(embed_dims, embed_dims) if project_scale == 2 else nn.Identity(),
            nn.Conv3d(embed_dims, num_classes, kernel_size=1))
        
        #### memory bank for saving region-tokens ################################
        num_keys, num_tokens = 3, 1024
        self.MB = MemoryBank(num_tokens=num_tokens, num_keys=num_keys, token_dim=embed_dims)
        self.diffuser = ASPP(embed_dims, (1, 3, 5))
        # self.diffuser = DeformConv3d(in_channels=embed_dims, out_channels=embed_dims)
        ################################################################################

    def generate_visible_mask(self, vol_pts):
        complete_x = vol_pts[0,:,0].clamp(0, self.scene_shape[0]-1)
        complete_y = vol_pts[0,:,1].clamp(0, self.scene_shape[1]-1)
        complete_z = vol_pts[0,:,2].clamp(0, self.scene_shape[2]-1)
        complete_mask = torch.zeros(self.scene_shape)
        complete_mask[complete_x, complete_y, complete_z] = 1
        complete_mask[:,:,2:] = 1
        return complete_mask[:,:,0]

    def mask_from_pts(self, shape, points, kernel_size=15):
        # Scale points to fit in the mask
        points = (127. * points).to(int)
        
        # Initialize the binary mask
        mask = torch.zeros(shape[0], shape[1], dtype=torch.float32)
        mask[points[:, 0], points[:, 1]] = 1
        mask_4d = mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, height, width]

        # Create a circular kernel
        radius = kernel_size // 2
        y, x = torch.meshgrid(torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1))
        circular_kernel = (x**2 + y**2 <= radius**2).float()
        circular_kernel = circular_kernel / circular_kernel.sum()  # Normalize the kernel

        # Convert the kernel to 4D for convolution: [batch_size, channels, height, width]
        circular_kernel = circular_kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, kernel_size, kernel_size]

        # Apply the convolution for diffusion
        diffused_mask = F.conv2d(mask_4d, circular_kernel, padding=radius)

        # Remove the extra dimensions
        diffused_mask = diffused_mask.squeeze()

        # Binarize the diffused mask
        binary_diffused_mask = (diffused_mask > 0).float()

        return binary_diffused_mask[:shape[0], :shape[1]]
        
    def replace_tokens(self, scene_embed, incomplete_mask, replacement_tokens):
        # Find the indices of the ones in the binary mask
        indices = torch.nonzero(incomplete_mask, as_tuple=False)
        # Ensure the number of ones in the mask matches the number of replacement tokens
        assert indices.size(0) == replacement_tokens.size(0), "Number of ones in the mask must match number of replacement tokens"
        # Get the coordinates (x, y) from the indices
        x_coords = indices[:, 0]
        y_coords = indices[:, 1]
        
        # Replace the tokens at (x, y, 0) in the 3D feature tensor with the replacement tokens
        scene_embed[:, :, x_coords, y_coords, 0] = replacement_tokens.T
        return scene_embed
    
    def check_recomplete_num(self, mask):
    # Ensure the input mask is a torch tensor
        if not isinstance(mask, torch.Tensor):
            raise ValueError("Input mask should be a PyTorch tensor.")
        
        # Count the number of ones in the mask
        ones_count = torch.sum(mask).item()
        
        # If there are more than 512 ones, randomly select 512
        if ones_count > 512:
            # Get the indices of all the ones in the mask
            ones_indices = torch.nonzero(mask, as_tuple=False)
            
            # Randomly select 512 indices
            selected_indices = ones_indices[torch.randperm(ones_indices.size(0))[:512]]
            
            # Create a new mask of zeros
            new_mask = torch.zeros_like(mask)
            
            # Set the selected indices to one
            new_mask[selected_indices[:, 0], selected_indices[:, 1]] = 1
            
            return new_mask
        else:
            return mask
    
    def mask_to_coordinates(self, mask):
        # Ensure the input mask is a torch tensor
        if not isinstance(mask, torch.Tensor):
            raise ValueError("Input mask should be a PyTorch tensor.")
        
        # Ensure the mask has exactly 512 ones
        # if torch.sum(mask).item() != 512:
        #     raise ValueError("The mask should contain exactly 512 ones.")
        
        # Get the indices of all the ones in the mask
        ones_indices = torch.nonzero(mask, as_tuple=False).float()
        
        # Normalize the indices to be in the range [0, 1]
        normalized_indices = ones_indices / torch.tensor(mask.shape).float().to(mask.device)
        
        return normalized_indices
    
    @autocast(dtype=torch.float32)
    def forward(self, seq, frame_id, pred_insts, feats, pred_masks, depth, K, E, voxel_origin, projected_pix,
                fov_mask):
        inst_queries = pred_insts['queries']  # bs, n, c
        inst_pos = pred_insts.get('query_pos', None)
        bs = inst_queries.shape[0]

        if self.downsample_z != 1:
            projected_pix = interpolate_flatten(
                projected_pix, self.ori_scene_shape, self.scene_shape, mode='trilinear')
            fov_mask = interpolate_flatten(
                fov_mask, self.ori_scene_shape, self.scene_shape, mode='trilinear')
        vol_pts = pix2vox(
            self.image_grid,
            depth.unsqueeze(1),
            K,
            E,
            voxel_origin,
            self.voxel_size,
            downsample_z=self.downsample_z).long()
        
        ref_2d = pred_insts['pred_pts'].unsqueeze(2).expand(-1, -1, len(feats), -1)
        ref_3d = self.generate_vol_ref_pts_from_pts(pred_insts['pred_pts'], vol_pts).unsqueeze(2)
        #### points initialization 
        self.cur_pts = ref_3d[0,:,0,:2]
        ref_pix = (torch.flip(projected_pix, dims=[-1]) + 0.5) / torch.tensor(
            self.image_shape).to(projected_pix)
        ref_pix = torch.flip(ref_pix, dims=[-1])
        ref_vox = nchw_to_nlc(self.voxel_grid.unsqueeze(0)).unsqueeze(2)

        #### generate incomplete BEV binary mask #######################################
        complete_bev_mask = self.generate_visible_mask(vol_pts).to(self.scene_embed.weight.device)
        incomplete_bev_mask = 1-complete_bev_mask

        scene_embed = self.scene_embed.weight.repeat(bs, 1, 1)
        scene_pos = self.scene_pos().repeat(bs, 1, 1)
        scene_embed = self.voxel_proposal(scene_embed, feats, scene_pos, vol_pts, ref_pix)
        scene_pos = nlc_to_nchw(scene_pos, self.scene_shape)
        outs = []
        for i, layer in enumerate(self.layers):
            scene_embed, inst_queries = layer(scene_embed, inst_queries, feats, scene_pos, inst_pos,
                                              ref_2d, ref_3d, ref_vox, fov_mask)

            if i == 2:

                #### scene recompletion 1st
                # mask_from_pts = self.mask_from_pts(self.scene_shape, self.cur_pts, kernel_size=10).to(scene_embed.device)
                # recomplete_mask = (incomplete_bev_mask==1)&(mask_from_pts==1)
                # recomplete_mask_512 = self.check_recomplete_num(recomplete_mask)
                # replace_tokens = self.MB.query_tokens(recomplete_mask_512, inst_queries[0], ref_3d[0,:,0,:])
                # scene_embed = scene_embed + self.replace_tokens(scene_embed, recomplete_mask_512, replace_tokens)
                
                #### scene recompletion 2nd
                # recomplete_pts = self.mask_to_coordinates(recomplete_mask_512)
                # self.cur_pts = torch.cat((self.cur_pts, recomplete_pts), 0)
                # mask_from_pts = self.mask_from_pts(self.scene_shape, self.cur_pts, kernel_size=20).to(scene_embed.device)
                # recomplete_mask_2 = (incomplete_bev_mask==1)&(mask_from_pts==1)
                # recomplete_mask_2 = torch.logical_xor(recomplete_mask_2, recomplete_mask)
                # recomplete_mask_512_2 = self.check_recomplete_num(recomplete_mask)
                # recomplete_pts = torch.cat((recomplete_pts, torch.zeros(recomplete_pts.shape[0], 1).to(recomplete_pts.device)), dim=1)
                # replace_tokens = self.MB.query_tokens(recomplete_mask_512_2, 
                #                                       torch.cat((inst_queries[0], replace_tokens), 0), 
                #                                       torch.cat((ref_3d[0,:,0,:], recomplete_pts), 0 ))
                # scene_embed = scene_embed + self.diffuser(self.replace_tokens(scene_embed, recomplete_mask_512_2, replace_tokens))
                #### ASPP
                scene_embed = self.aspp(scene_embed)
                # scene_embed = self.diffuser(scene_embed)
                # print(scene_embed.shape)

                if self.training:
                    #### memory updating 
                    self.MB.update_memory_bank(inst_queries[0].data, ref_3d[0,:,0,:])
                    
            if self.training or i == len(self.layers) - 1:
                outs.append(self.cls_head(scene_embed))
        return outs, scene_embed, complete_bev_mask

    def generate_vol_ref_pts_from_masks(self, pred_boxes, pred_masks, vol_pts):
        pred_boxes *= torch.tensor((self.image_shape + self.image_shape)[::-1]).to(pred_boxes)
        pred_pts = pred_boxes[..., :2].int()
        cx, cy, w, h = pred_boxes.split((1, 1, 1, 1), dim=-1)
        pred_boxes = torch.cat([(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)],
                               dim=-1).int()
        pred_boxes[0::2] = pred_boxes[0::2].clamp(0, self.image_shape[1] - 1)
        pred_boxes[1::2] = pred_boxes[1::2].clamp(1, self.image_shape[1] - 1)

        pred_masks = F.interpolate(
            pred_masks.float(), self.image_shape, mode='bilinear').to(pred_masks.dtype)
        bs, n = pred_masks.shape[:2]

        for b, i in product(range(bs), range(n)):
            if pred_masks[b, i].sum().item() != 0:
                continue
            boxes = pred_boxes[b, i]
            pred_masks[b, i, boxes[1]:boxes[3], boxes[0]:boxes[2]] = True
            if pred_masks[b, i].sum().item() != 0:
                continue
            pred_masks[b, i, pred_pts[b, i, 1], pred_pts[b, i, 0]] = True
        pred_masks = pred_masks.flatten(2).unsqueeze(-1).to(vol_pts)  # bs, n, hw, 1
        vol_pts = vol_pts.unsqueeze(1) * pred_masks  # bs, n, hw, 3
        vol_pts = vol_pts.sum(dim=2) / pred_masks.sum(dim=2) / torch.tensor(
            self.scene_shape).to(vol_pts)
        return vol_pts.clamp(0, 1)

    def generate_vol_ref_pts_from_pts(self, pred_pts, vol_pts):
        pred_pts = pred_pts * torch.tensor(self.image_shape[::-1]).to(pred_pts)
        pred_pts = pred_pts.long()
        pred_pts = pred_pts[..., 1] * self.image_shape[1] + pred_pts[..., 0]
        assert pred_pts.size(0) == 1
        ref_pts = vol_pts[:, pred_pts.squeeze()]
        ref_pts = ref_pts / (torch.tensor(self.scene_shape) - 1).to(pred_pts)
        return ref_pts.clamp(0, 1)
