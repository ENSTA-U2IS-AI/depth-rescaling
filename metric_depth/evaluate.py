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
import time

import torch
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters)



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


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    if config.rescale_with == 'stereo':
        metrics_ref = RunningAverageDict()
        
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth = sample['image'], sample['depth']
        image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        
        pred = infer(model, image, dataset=sample['dataset'][0], rescale_with=config.rescale_with, focal=focal)

        # Save image, depth, pred for visualization
        if "save_images" in config and config.save_images:
            import os
            # print("Saving images ...")
            from PIL import Image
            import torchvision.transforms as transforms
            from zoedepth.utils.misc import colorize

            os.makedirs(config.save_images, exist_ok=True)
            # def save_image(img, path):
            d = colorize(depth.squeeze().cpu().numpy(), 0, 10)
            p = colorize(pred.squeeze().cpu().numpy(), 0, 10)
            im = transforms.ToPILImage()(image.squeeze().cpu())
            im.save(os.path.join(config.save_images, f"{i}_img.png"))
            Image.fromarray(d).save(os.path.join(config.save_images, f"{i}_depth.png"))
            Image.fromarray(p).save(os.path.join(config.save_images, f"{i}_pred.png"))
            
        if config.rescale_with == 'stereo':
            metrics.update(compute_metrics(depth, pred, config=config, stereo_depth=sample['stereo_depth'], stereo_valid=sample['stereo_valid'])) 
            metric_sample = compute_metrics(depth, sample['stereo_filled'].squeeze().unsqueeze(0).unsqueeze(0).cuda(), is_stereo_ref=True, config=config)
            if not np.isnan(metric_sample['a1']): # check if all metrics are NaN (only 1a metric is tested)
                metrics_ref.update(metric_sample)
                
        elif config.rescale_with == 'sfm':
            
            if sample['valid_pose'][0]:
                metric_sample = compute_metrics(depth, pred, config=config, sfm_depth=sample['sfm_depth'], sfm_pts=sample['sfm_pts'], confidence=sample['confidence'])
                metrics.update(metric_sample)
                    
        else:
            metric_sample = compute_metrics(depth, pred, config=config, rescale_with=config.rescale_with)
            metrics.update(metric_sample)
            
    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    if config.rescale_with == 'stereo':
        metrics_ref = {k: r(v) for k, v in metrics_ref.get_value().items()}
        return metrics, metrics_ref
    return metrics

def main(config):
    model = build_model(config)
    test_loader = DepthDataLoader(config, 'online_eval').data
    model = model.cuda()
    if config.rescale_with == 'stereo':
        metrics, metrics_ref = evaluate(model, test_loader, config)
        print(f"{colors.fg.red}")
        print(metrics_ref)
        print(f"{colors.reset}")
    else:
        metrics = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return metrics


def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):

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
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")
    parser.add_argument("-r", "--rescale_with", type=str, help="Name of the reference to rescale the disparity", choices=['lidar', 'stereo', 'sfm'])
    parser.add_argument("--matching_with", type=str, help="Name of the reference to rescale the disparity", choices=['sift', 'omniglue'])
    parser.add_argument("--num_beams", type=int, help="Number of beams to keep in the lidar")
    parser.add_argument("--max_stereo", default=80, type=float, help="Max stereo depth to consider for rescaling")
    parser.add_argument("-t", "--translation_threshold", type=float, help="Min translation")
    parser.add_argument("--path_to_keypoints", type=str, help="Path to keypoints files (for sfm with OmniGlue)")
    
    
    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)
    
    overwrite_kwargs['rescale_with'] = args.rescale_with
    overwrite_kwargs['matching_with'] = args.matching_with
    overwrite_kwargs['num_beams'] = args.num_beams
    overwrite_kwargs['translation_threshold'] = args.translation_threshold
    overwrite_kwargs['max_stereo'] = args.max_stereo
    overwrite_kwargs['path_to_keypoints'] = args.path_to_keypoints

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
