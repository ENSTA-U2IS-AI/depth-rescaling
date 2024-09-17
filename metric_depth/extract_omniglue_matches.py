# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Rémi Marsal

# This file is partly inspired from Depth AnyThing (https://github.com/LiheYoung/Depth-Anything/blob/main/metric_depth/evaluate.py); author: Shariq Farooq Bhat

import argparse
from pprint import pprint
import os
import numpy as np

import torch
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.data_extract_omniglue_matches_kitti import OmniGlueKITTIDataLoader
from zoedepth.data.data_extract_omniglue_matches_ddad import get_omniglue_ddad_loader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR

import omniglue


@torch.no_grad()
def infer(model, images, rescale_with, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            if rescale_with is not None:
                pred = pred['rel_depth'].unsqueeze(1)
            else:
                pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred
    
    preds = model(torch.cat([images, torch.flip(images, [3])]), **kwargs)
    preds = get_depth_from_prediction(preds)
    mean_pred = 0.5 * (preds[:1] + torch.flip(preds[1:], [3]))
    return mean_pred
        

def main(config):
    og = omniglue.OmniGlue(
        og_export='./models/og_export',
        sp_export='./models/sp_v6',
        dino_export='./models/dinov2_vitb14_pretrain.pth',
        )
    
    if config.dataset == 'kitti':
        test_loader = OmniGlueKITTIDataLoader(config, 'online_eval').data
    elif config.dataset == 'ddad':
        test_loader = get_omniglue_ddad_loader(config.ddad_root, resize_shape=(
                    352, 1216), batch_size=1, num_workers=1)
    
    os.makedirs(config.save_path, exist_ok=True)
    
    for _, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        image_path = sample['image_path'][0]
        if sample['valid_prev'][0]:
            im0 = sample['im0'].cpu().squeeze().numpy()
            im1 = sample['im1'].cpu().squeeze().numpy()
            
            if config.dataset == 'kitti':
                matches_file = os.path.join(config.save_path, os.path.splitext(sample['image_path'][0].replace('/', '_'))[0] + '.npz')
            elif config.dataset == 'ddad':
                sequence = image_path.split('/')[-4]
                directory = os.path.join(config.save_path, sequence)
                os.makedirs(directory, exist_ok=True)
                matches_file = os.path.join(directory, os.path.splitext(os.path.basename(image_path))[0] + '.npz')
            match_kp0s, match_kp1s, match_confidences = og.FindMatches(im0, im1)
            
            np.savez(matches_file, match_kp0s=match_kp0s, match_kp1s=match_kp1s, match_confidences=match_confidences)
        
        else:
            print('No image before', sample['image_path'][0])


def eval_model(model_name, pretrained_resource=None, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Name of the model to evaluate", default='zoedepth')
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")
    parser.add_argument("--save_path", type=str, help="Path to keypoints files (for sfm with OmniGlue)")

    
    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)
    
    overwrite_kwargs['save_path'] = args.save_path
    
    if "ALL_INDOOR" in args.dataset:
        datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset:
        datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset:
        datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource,
                    dataset=dataset, **overwrite_kwargs)
