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
    label_map = np.array([KITTI_LABEL_MAP[i] for i in range(len(KITTI_LABEL_MAP))], dtype=np.int32)

    with torch.no_grad():
        
        false_negative_heatmap = np.zeros((256, 256), dtype=int)
        #### initialization for false negative ratio 
        class_show_cnts = np.ones(20)
        false_negative_accumulate = np.zeros(20)
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
                fov_mask = batch_inputs['fov_mask_1'][i].detach().cpu().numpy()
                target = targets['target'][0].detach().cpu().numpy().astype(np.int32)
                #### visualize the false negative mask#############################################
                false_negative_mask = (target!=0)&(target!=255)&(pred==0)
                all_mask = (target!=0)&(target!=255)
                masked_values = target[false_negative_mask]
                all_values = target[all_mask]
                false_negative_class_counts = np.bincount(masked_values, minlength=20)
                all_class_counts = np.bincount(all_values, minlength=20)
                # false_negative_accumulate += np.nan_to_num(false_negative_class_counts/all_class_counts, nan=0.0)
                false_negative_accumulate += np.nan_to_num(false_negative_class_counts, nan=0.0)
                for class_idx in range(20):
                    if np.any(all_values == class_idx):
                        class_show_cnts[class_idx] += 1
                
                # from 3D to BEV
                false_negative_mask = np.any(false_negative_mask, axis=2).astype(int) 
                # false_negative_heatmap = false_negative_heatmap + false_negative_mask
                colormap = plt.cm.viridis

                # Create a colored visualization of the binary mask
                colored_mask = colormap(false_negative_mask)

                # Save the colored image
                seq = batch_inputs['sequence'][0]
                frame_id = batch_inputs['frame_id'][0]
                file_name = f'./visualization/semkitti-false-negative-mask-colored/{str(seq)}-{str(frame_id).zfill(6)}.png'
                plt.imsave(file_name, colored_mask)
                print(file_name)
                #### Adjust the visualization #####################################################
                
                # false_negative_scene = np.where(false_negative_mask, target, 0)
                # false_negative_scene = false_negative_scene.reshape(-1)
                # false_negative_scene = label_map[false_negative_scene].astype(np.uint16)
                
                # pred[:100,:,:] = 0
                # pred = pred.reshape(-1)
                # pred[fov_mask==False] = 0 
                # pred = label_map[pred].astype(np.uint16)
                ###################################################################################
                # save_dir = osp.join(output_dir, f"false-negative/sequences/{batch_inputs['sequence'][i]}/predictions")
                # file_path = osp.join(save_dir, f"{batch_inputs['frame_id'][i]}.label")
                # os.makedirs(save_dir, exist_ok=True)
                # false_negative_scene.tofile(file_path)
                # print('saved to', file_path)
        false_negative_ratio = false_negative_accumulate/false_negative_accumulate.max()
        plt.figure(figsize=(16, 12))
        bars = plt.bar(range(len(false_negative_ratio)), false_negative_ratio, color='skyblue', edgecolor='black')

        # Add labels and title
        plt.xlabel('Class', fontsize=14)
        plt.ylabel('Ratio', fontsize=14)
        plt.title('Ratios by Class', fontsize=16)
        plt.xticks(range(len(false_negative_ratio)), class_names, rotation=45, ha='right', fontsize=12)
        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

        # Add gridlines for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('./visualization/semantickitti-false-negative.png')
        # plt.figure(figsize=(10, 8))
        # false_negative_heatmap = false_negative_heatmap / false_negative_heatmap.max()
        # plt.imshow(false_negative_heatmap, cmap='hot', interpolation='nearest')
        # plt.colorbar(label='Accumulation count')
        # plt.title('Accumulated BEV Mask Heatmap')
        # plt.xlabel('X')
        # plt.ylabel('Y')

        # Save the heatmap as an image file
        # plt.savefig('./visualization/accumulated_bev_mask_heatmap.png')
            # emp_conf_dis = ratio_sum/frame_cnt
                        
            # plt.bar(np.linspace(0, 1, len(emp_conf_dis)), emp_conf_dis, width=0.02)  # Adjust the width as needed
            # plt.title('Histogram of Tensor Values (Normalized)')
            # plt.xlabel('Value')
            # plt.ylabel('Ratio')
            # plt.savefig('visualize/false_nagative_histogram.png')

if __name__ == '__main__':
    main()
