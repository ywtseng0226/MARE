import os
import os.path as osp
import time
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
import matplotlib.pyplot as plt

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks

class_names = (
    'empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
    'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
)

KITTI360_LABEL_MAP = {
    0: 0  ,    # "unlabeled", and others ignored
    1: 10 ,    # "car"
    2: 11 ,    # "bicycle"
    3: 15 ,    # "motorcycle"
    4: 18 ,    # "truck"
    5: 20 ,    # "other-vehicle"
    6: 30 ,    # "person"
    7: 40 ,    # "road"
    8: 44 ,    # "parking"
    9: 48,    # "sidewalk"
    10: 49,    # "other-ground"
    11: 50,    # "building"
    12: 51,    # "fence"
    13: 70,    # "vegetation"
    14: 72,    # "terrain"
    15: 80,    # "pole"
    16: 81,    # "traffic-sign"
    17: 52,    # "other-structure"
    18: 99,    # "other-object"
}


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, _ = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    data_loader = dls[-1]
    output_dir = osp.join('outputs', cfg.data.datasets.type)

    if cfg.get('ckpt_path'):
        model = LitModule.load_from_checkpoint(cfg.ckpt_path, **cfg, meta_info=meta_info)
    else:
        import warnings
        warnings.warn('\033[31;1m{}\033[0m'.format('No checkpoint path is provided'))
        model = LitModule(**cfg, meta_info=meta_info)
    model.cuda()
    model.eval()

    # assert cfg.data.datasets.type == 'SemanticKITTI'
    print(f'Current using dataset: {cfg.data.datasets.type} ..........')
    label_map = np.array([KITTI360_LABEL_MAP[i] for i in range(len(KITTI360_LABEL_MAP))], dtype=np.int32)

    with torch.no_grad():
        
        for batch_inputs, targets in track(data_loader):
            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs[key] = batch_inputs[key].cuda()
            for key in targets:
                if isinstance(targets[key], torch.Tensor):
                    targets[key] = targets[key].cuda()
            # starttime = time.time()
            outputs = model(batch_inputs)
            # endtime = time.time()
            # print(f'inference time: {endtime-starttime}')
            preds = torch.softmax(outputs['ssc_logits'], dim=1).detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            
            for i in range(preds.shape[0]):
                # pred = label_map[preds[i].reshape(-1)].astype(np.uint16)
                pred = preds[i]
                target = targets['target'][0].detach().cpu().numpy().astype(np.int32)
                invalud_mask = target==255
                pred[:,:,2:][invalud_mask[:,:,2:]] = 0
                pred = pred.reshape(-1)
                # pred[fov_mask==False] = 0 
                pred = label_map[pred].astype(np.uint16)
                #### save the visualization
                save_dir = osp.join(output_dir,
                                    f"teseting/sequences/{batch_inputs['sequence'][i]}/predictions")
                file_path = osp.join(save_dir, f"{batch_inputs['frame_id'][i]}.label")
                os.makedirs(save_dir, exist_ok=True)
                pred.tofile(file_path)
                print('saved to', file_path)

if __name__ == '__main__':
    main()
