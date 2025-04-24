import os
import os.path as osp
import hydra
import numpy as np
import torch
import time  # Import time module
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks, SSCMetrics
from sklearn.metrics import confusion_matrix

KITTI_LABEL_MAP = {
    0: 0,  # unlabeled
    1: 10,  # car
    2: 11,  # bicycle
    3: 15,  # motorcycle
    4: 18,  # truck
    5: 20,  # other-vehicle
    6: 30,  # person
    7: 31,  # bicyclist
    8: 32,  # motorcyclist
    9: 40,  # road
    10: 44,  # parking
    11: 48,  # sidewalk
    12: 49,  # other-ground
    13: 50,  # building
    14: 51,  # fence
    15: 70,  # vegetation
    16: 71,  # trunk
    17: 72,  # terrain
    18: 80,  # pole
    19: 81,  # traffic-sign
}

KITTI360_LABEL_MAP = {
    0: 0,  # "unlabeled", and others ignored
    1: 10,  # "car"
    2: 11,  # "bicycle"
    3: 15,  # "motorcycle"
    4: 18,  # "truck"
    5: 20,  # "other-vehicle"
    6: 30,  # "person"
    7: 40,  # "road"
    8: 44,  # "parking"
    9: 48,  # "sidewalk"
    10: 49,  # "other-ground"
    11: 50,  # "building"
    12: 51,  # "fence"
    13: 70,  # "vegetation"
    14: 72,  # "terrain"
    15: 80,  # "pole"
    16: 81,  # "traffic-sign"
    17: 52,  # "other-structure"
    18: 99,  # "other-object"
}

# Count model parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

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

    count_parameters(model)
    model.cuda()
    model.eval()

    print(f'Current using dataset: {cfg.data.datasets.type} ..........')
    evaluator = SSCMetrics(num_classes=19).cuda()

    # Start timing
    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in track(data_loader):
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].cuda()
            for key in targets:
                if isinstance(targets[key], torch.Tensor):
                    targets[key] = targets[key].cuda()

            outputs = model(inputs)
            visible_mask = outputs["visible_mask"].unsqueeze(0).unsqueeze(-1)  # Shape: [1, 256, 256, 1]
            visible_mask = visible_mask.expand(-1, -1, -1, 32)  # Shape: [1, 256, 256, 32]
            targets['target'][visible_mask == 1] = 255.
            evaluator.update(outputs, targets)

    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_batch = total_time / len(data_loader)
    # Compute performance metrics
    performance = evaluator.compute()
    for metric in performance.keys():
        print(f'{metric}: {performance[metric]}')
    # Print inference time information
    print(f'Total inference time: {total_time:.2f} seconds')
    print(f'Average time per batch: {avg_time_per_batch:.4f} seconds')
            
if __name__ == '__main__':
    main()
