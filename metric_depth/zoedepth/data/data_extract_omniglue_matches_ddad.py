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

# This file is partly inspired from Depth AnyThing (https://github.com/LiheYoung/Depth-Anything/blob/main/metric_depth/zoedepth/data/data_mono.py); author: Shariq Farooq Bhat

import os
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2

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


class OmniGlueDDADDataset(Dataset):
    def __init__(self, data_dir_root, resize_shape, **kwargs):
        import glob

        # image paths are of the form <data_dir_root>/{outleft, depthmap}/*.png
        
        # self.image_files = glob.glob(os.path.join(data_dir_root, '*.png'))
        # self.depth_files = [r.replace("_rgb.png", "_depth.npy")
        #                     for r in self.image_files]
        self.image_files, self.depth_files = [], []
        with open('./train_test_inputs/val.txt', 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            self.image_files.append(line.split(' ')[0])
            self.depth_files.append(line.split(' ')[1])
        
        self.transform = ToTensor(resize_shape)

        path = './train_test_inputs/ddad.json'

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
            
        self.current_idx = 0
            
            
    def __getitem__(self, idx):

        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(depth_path), dtype=np.float32) / 256 #np.load(depth_path)  # meters
        
        timestamp = os.path.basename(os.path.splitext(image_path)[0])
        data_sample = self.dgp[self.current_idx]
        valid_data = timestamp == str(self.dgp.dataset[self.current_idx][1][0]['timestamp'])
        
        if not valid_data:
            pass
            # print('no previous image for depth file', depth_path)
        else:
            self.current_idx += 1
            
        # im0 = np.asarray(data_sample[0][0]['rgb'])[..., :3]
        # im1 = np.asarray(data_sample[1][0]['rgb'])[..., :3]
        im1 = np.array(data_sample['rgb'])
        im0 = np.array(data_sample['rgb_context'][0])

        # depth[depth > 8] = -1
        depth = depth[..., None]

        sample = dict(image=image, depth=depth)
        sample = self.transform(sample)
            
        sample['image_path'] = image_path
        sample['depth_path'] = depth_path
        
        sample['im1'] = im1
        sample['im0'] = im0
        sample['timestamp'] = timestamp
        
        sample['valid_prev'] = np.array([valid_data])

        return sample

    def __len__(self):
        return len(self.image_files)


def get_omniglue_ddad_loader(data_dir_root, resize_shape, batch_size=1, **kwargs):
    dataset = OmniGlueDDADDataset(data_dir_root, resize_shape, **kwargs)
    return DataLoader(dataset, batch_size, **kwargs)
