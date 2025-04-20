import torch.nn as nn
import torch
import numpy as np 
import matplotlib.pyplot as plt

from ... import build_from_configs
from .. import encoders
from ..decoders import MAREDecoder
from ..losses import ce_ssc_loss, frustum_proportion_loss, geo_scal_loss, sem_scal_loss


class MARE(nn.Module):

    def __init__(
        self,
        encoder,
        embed_dims,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        num_layers=3,
        image_shape=(370, 1220),
        voxel_size=0.2,
        downsample_z=2,
        class_weights=None,
        criterions=None,
        **kwargs,
    ):
        super().__init__()
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions

        self.encoder = build_from_configs(
            encoders, encoder, embed_dims=embed_dims, scales=view_scales)
        self.decoder = MAREDecoder(
            embed_dims,
            num_classes,
            num_layers=num_layers,
            num_levels=len(view_scales),
            scene_shape=scene_size,
            project_scale=volume_scale,
            image_shape=image_shape,
            voxel_size=voxel_size,
            downsample_z=downsample_z)

    def forward(self, inputs):
        pred_insts = self.encoder(inputs['img'])
        pred_masks = pred_insts.pop('pred_masks', None)
        feats = pred_insts.pop('feats')
        depth, K, E, voxel_origin, projected_pix, fov_mask = list(
            map(lambda k: inputs[k],
                ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
                 f'fov_mask_{self.volume_scale}')))

        outs, scene_embed, visible_bev_mask = self.decoder(inputs['sequence'][0], inputs['frame_id'][0], pred_insts, feats, pred_masks, 
                                         depth, K, E, voxel_origin, projected_pix, fov_mask)
        
        visible_bev_mask = visible_bev_mask.unsqueeze(0).unsqueeze(0)
        visible_bev_mask = torch.nn.functional.interpolate(visible_bev_mask, size=(256, 256), mode='bilinear', align_corners=False)
        visible_bev_mask = (visible_bev_mask > 0.5).float()
        visible_bev_mask = visible_bev_mask.squeeze(0).squeeze(0)

        return {'ssc_logits': outs[-1], 'aux_outputs': outs, 'visible_mask':visible_bev_mask}

    def loss(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'frustum': frustum_proportion_loss
        }

        target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
        losses = {}
        if 'aux_outputs' in preds:
            for i, pred in enumerate(preds['aux_outputs']):
                scale = 1 if i == len(preds['aux_outputs']) - 1 else 0.5
                for loss in self.criterions:
                    losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
                        'ssc_logits': pred
                    }, target) * scale
        else:
            for loss in self.criterions:
                losses['loss_' + loss] = loss_map[loss](preds, target)
        # for key in losses:
        #     if '_ce_' in key: print(key, '---', losses[key].item(), end=' ')
        # print()
        return losses
