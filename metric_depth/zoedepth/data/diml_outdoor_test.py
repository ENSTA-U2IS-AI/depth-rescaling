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

# This file is partly inspired from Depth AnyThing (https://github.com/LiheYoung/Depth-Anything/blob/main/metric_depth/zoedepth/data/diml_outdoor_test.py); author: Shariq Farooq Bhat

import os

import numpy as np
import torch
from PIL import Image
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from zoedepth.utils.rescaling import StereoSGBM


class ToTensor(object):
    def __init__(self):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x : x

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        return {'image': image, 'depth': depth, 'dataset': "diml_outdoor"}

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


class DIML_Outdoor(Dataset):
    def __init__(self, data_dir_root, config):
        import glob

        # image paths are of the form <data_dir_root>/{outleft, depthmap}/*.png
        self.image_files = glob.glob(os.path.join(
            data_dir_root, 'outleft', '*.png'))
        self.depth_files = [r.replace("outleft", "depthmap")
                            for r in self.image_files]
        self.image_right_files = [r.replace("outleft", "outright")
                            for r in self.image_files]
        self.stereo_files = [r.replace("outleft", "stereo") + '.npz'
                            for r in self.image_files]
        self.transform = ToTensor()
        self.config = config
            
        if self.config.rescale_with == 'stereo':
            self.stereo_matcher = StereoSGBM()

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(depth_path),
                           dtype='uint16') / 1000.0  # mm to meters

        # depth[depth > 8] = -1
        depth = depth[..., None]

        sample = dict(image=image, depth=depth, dataset="diml_outdoor")
            
        sample['image_path'] = image_path
        sample['depth_path'] = depth_path
        
        sample = self.transform(sample)

        if self.config.rescale_with == 'stereo':
            
            image_right_path = self.image_right_files[idx]
            image_right = np.asarray(Image.open(image_right_path), dtype=np.uint8)
            image_left = np.asarray(Image.open(image_path), dtype=np.uint8)
            stereo_disp, stereo_valid = self.stereo_matcher.get_disp(image_left, image_right)
            
            stereo_valid = stereo_disp > 0
            stereo_depth =  4 * 12 / stereo_disp
            stereo_valid = np.logical_and(np.logical_and(stereo_valid, 2 < stereo_depth), stereo_depth < 80)
            
            sample['stereo_valid'] = stereo_valid
            sample['stereo_depth'] = stereo_depth
            
            stereo_filled = cv2.inpaint(stereo_disp.astype(np.float32), (1 - stereo_valid).astype(np.uint8), 3, cv2.INPAINT_TELEA)
            sample['stereo_filled'] = 4 * 12 / stereo_filled

        # return sample
        return sample

    def __len__(self):
        return len(self.image_files)


def get_diml_outdoor_loader(data_dir_root, config, batch_size=1, **kwargs):
    dataset = DIML_Outdoor(data_dir_root, config)
    return DataLoader(dataset, batch_size, **kwargs)

# get_diml_outdoor_loader(data_dir_root="datasets/diml/outdoor/test/HR")
# get_diml_outdoor_loader(data_dir_root="datasets/diml/outdoor/test/LR")
