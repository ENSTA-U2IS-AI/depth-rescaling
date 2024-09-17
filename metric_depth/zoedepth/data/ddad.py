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

# This file is partly inspired from Depth AnyThing (https://github.com/LiheYoung/Depth-Anything/blob/main/metric_depth/zoedepth/data/ddad.py); author: Shariq Farooq Bhat

import os
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from zoedepth.utils.rescaling import StereoSGBM, SFM_ddad, get_sift_matching_keypoints

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.utils.camera import Camera, generate_depth_map
from dgp.utils.pose import Pose

from zoedepth.utils.packnet_dgp_data_utils import DGPDataset

class ToTensor(object):
    def __init__(self, resize_shape):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x : x
        self.resize = transforms.Resize(resize_shape)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        image = self.resize(image)

        return {'image': image, 'depth': depth, 'dataset': "ddad"}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class DDAD(Dataset):
    def __init__(self, data_dir_root, resize_shape, config, **kwargs):
        import glob
        
        self.config = config

        # image paths are of the form <data_dir_root>/{outleft, depthmap}/*.png
        
        # self.image_files = glob.glob(os.path.join(data_dir_root, '*.png'))
        # self.depth_files = [r.replace("_rgb.png", "_depth.npy")
        #                     for r in self.image_files]
        self.image_files, self.depth_files = [], []
        with open('./train_test_inputs/val_ddad.txt', 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            self.image_files.append(line.split(' ')[0])
            self.depth_files.append(line.split(' ')[1])
        
        self.transform = ToTensor(resize_shape)
        
        self.rescale_with = config.rescale_with
        if self.rescale_with == 'sfm':

            path = '/home/remi/data/DDAD/ddad_train_val/ddad.json'

            split = 'val'
            
            dataset_args = {
                    'back_context': 1,
                    'forward_context': 0,
                    'data_transform': None
                }

            dataset_args_i = {
                'depth_type': None, # depth_type if 'gt_depth' in requirements else None,
                'input_depth_type': None, #input_depth_type if 'gt_depth' in requirements else None,
                'with_pose': True, #'gt_pose' in requirements,
            }

            self.dgp = DGPDataset(
                            path, split,
                            **dataset_args, **dataset_args_i,
                            cameras=['camera_01']
                        )
            self.sfm = SFM_ddad(config.translation_threshold)

        self.matching_with = config.matching_with
            
        self.current_idx = 0
            
            
    def __getitem__(self, idx):

        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(depth_path), dtype=np.float32) / 256 #np.load(depth_path)  # meters
        
        if self.rescale_with == 'sfm':
            timestamp = os.path.basename(os.path.splitext(image_path)[0])
            data_sample = self.dgp[self.current_idx]
            valid_data = timestamp == str(self.dgp.dataset[self.current_idx][1][0]['timestamp'])
            
            if not valid_data:
                # pass
                print('no previous image for depth file', depth_path)
            else:
                self.current_idx += 1
                
                im1 = np.array(data_sample['rgb'])
                im0 = np.array(data_sample['rgb_context'][0])  
                
                if self.matching_with == 'sift' and valid_data:

                    pts0, pts1 = get_sift_matching_keypoints(im0, im1)
                    
                    confidence = np.ones_like(pts1[:, 0]).astype(np.float32)
                    
                elif self.matching_with == 'omniglue' and valid_data:
                    
                    sequence = image_path.split('/')[-4]
                    directory = os.path.join(self.config.path_to_keypoints, sequence)
                    matches_file = os.path.join(directory, os.path.splitext(os.path.basename(image_path))[0] + '.npz')
                    
                    matches = np.load(matches_file)
                    pts0 = matches['match_kp0s'].astype(np.float32)
                    pts1 = matches['match_kp1s'].astype(np.float32)
                    confidence = matches['match_confidences'].astype(np.float32)

                sfm_depth, valid_pose = self.sfm(data_sample, pts0, pts1)

        # depth[depth > 8] = -1
        depth = depth[..., None]

        sample = dict(image=image, depth=depth)
        sample = self.transform(sample)
            
        sample['image_path'] = image_path
        sample['depth_path'] = depth_path
        
        if self.rescale_with == 'sfm' and valid_data:
            sample['sfm_pts'] = pts1
            sample['sfm_depth'] = sfm_depth
            sample['valid_pose'] = np.array([valid_pose and valid_data])
            sample['confidence'] = confidence
        else:
            sample['valid_pose'] = np.array([False])

        return sample

    def __len__(self):
        return len(self.image_files)


def get_ddad_loader(data_dir_root, resize_shape, config, batch_size=1, **kwargs):
    dataset = DDAD(data_dir_root, resize_shape, config, **kwargs)
    return DataLoader(dataset, batch_size, **kwargs)
