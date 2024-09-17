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

from typing import Any
import os
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor, LinearRegression
from skimage.color import rgb2gray

from zoedepth.utils.packnet_kitti_data_utils import pose_from_oxts_packet, read_calib_file, transform_from_rot_trans

# Cameras from the stero pair (left is the origin)
IMAGE_FOLDER = {
    'left': 'image_02',
    'right': 'image_03',
}
# Name of different calibration files
CALIB_FILE = {
    'cam2cam': 'calib_cam_to_cam.txt',
    'velo2cam': 'calib_velo_to_cam.txt',
    'imu2velo': 'calib_imu_to_velo.txt',
}
PNG_DEPTH_DATASETS = ['groundtruth']
OXTS_POSE_DATA = 'oxts'

    
def get_sift_matching_keypoints(im0, im1):
    sift = cv2.SIFT_create()
    im_0 = (rgb2gray(im0) * 255).astype(np.uint8)
    im_1 = (rgb2gray(im1) * 255).astype(np.uint8)

    kp0, des0 = sift.detectAndCompute(im_0, None)
    kp1, des1 = sift.detectAndCompute(im_1, None)
    
    # Match features using FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=2)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des0, k=2)

    # Extract location of good matches
    pts0 = np.float32([kp0[m.trainIdx].pt for m, n in matches]).reshape(-1, 2)
    pts1 = np.float32([kp1[m.queryIdx].pt for m, n in matches]).reshape(-1, 2)
    return pts0, pts1

class SFM_kitti:
    def __init__(self, translation_threshold) -> None:
        self.translation_threshold = translation_threshold
        self.rotation_threshold = 5.0
        
    @staticmethod
    def _get_parent_folder(image_file):
        """Get the parent folder from image_file."""
        return os.path.abspath(os.path.join(image_file, "../../../.."))
    
    @staticmethod
    def _get_intrinsics(image_file, calib_data):
        """Get intrinsics from the calib_data dictionary."""
        for cam in ['left', 'right']:
            # Check for both cameras, if found replace and return intrinsics
            if IMAGE_FOLDER[cam] in image_file:
                return np.reshape(calib_data[IMAGE_FOLDER[cam].replace('image', 'P_rect')], (3, 4))[:, :3]
            
    def _get_oxts_data(self, image_file):
        """Gets the oxts data from an image file."""
        oxts_file = self._get_oxts_file(image_file)
        oxts_data = np.loadtxt(oxts_file, delimiter=' ', skiprows=0)
        return oxts_data

    @staticmethod
    def _get_oxts_file(image_file):
        """Gets the oxts file from an image file."""
        # find oxts pose file
        for cam in ['left', 'right']:
            # Check for both cameras, if found replace and return file name
            if IMAGE_FOLDER[cam] in image_file:
                return image_file.replace(IMAGE_FOLDER[cam], OXTS_POSE_DATA).replace('.png', '.txt')
        # Something went wrong (invalid image file)
        raise ValueError('Invalid KITTI path for pose supervision.')
    
    @staticmethod
    def invert_pose_numpy(T):
        """Inverts a [4,4] np.array pose"""
        Tinv = np.copy(T)
        R, t = Tinv[:3, :3], Tinv[:3, 3]
        Tinv[:3, :3], Tinv[:3, 3] = R.T, - np.matmul(R.T, t)
        return Tinv
    
    def _get_imu2cam(self, image_file):
        parent_folder = self._get_parent_folder(image_file)

        cam2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['cam2cam']))
        imu2velo = read_calib_file(os.path.join(parent_folder, CALIB_FILE['imu2velo']))
        velo2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['velo2cam']))

        velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

        return cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

    def _get_pose(self, image_file):
        """Gets the pose information from an image file."""
        # Find origin frame in this sequence to determine scale & origin translation
        base, ext = os.path.splitext(os.path.basename(image_file))
        origin_frame = os.path.join(os.path.dirname(image_file), str(0).zfill(len(base)) + ext)
        # Get origin data
        origin_oxts_data = self._get_oxts_data(origin_frame)
        lat = origin_oxts_data[0]
        scale = np.cos(lat * np.pi / 180.)
        # Get origin pose
        origin_R, origin_t = pose_from_oxts_packet(origin_oxts_data, scale)
        origin_pose = transform_from_rot_trans(origin_R, origin_t)
        # Compute current pose
        oxts_data = self._get_oxts_data(image_file)
        R, t = pose_from_oxts_packet(oxts_data, scale)
        pose = transform_from_rot_trans(R, t)
        # Compute odometry pose
        imu2cam = self._get_imu2cam(image_file)
        odo_pose = (imu2cam @ np.linalg.inv(origin_pose) @
                    pose @ np.linalg.inv(imu2cam)).astype(np.float32)
        return odo_pose

    def check_pose_criterion(self, R, t):
        
        # Calculate the magnitude of the translation vector
        translation_magnitude = np.linalg.norm(t)

        # Calculate the rotation angle from the rotation matrix
        rotation_angle = np.arccos((np.trace(R) - 1) / 2) * (180 / np.pi)
        
        return translation_magnitude > self.translation_threshold or rotation_angle > self.rotation_threshold
    
    def __call__(self, file_im_0, pts0, file_im_1, pts1):
        parent_folder = self._get_parent_folder(file_im_1)
        calib_data = read_calib_file(os.path.join(parent_folder, CALIB_FILE['cam2cam']))
        K = self._get_intrinsics(file_im_1, calib_data)
        
        pose_1 = self._get_pose(file_im_1)
        pose_0 = self._get_pose(file_im_0)
        pose = self.invert_pose_numpy(pose_0) @ pose_1
        R, t = pose[:3, :3], pose[:3, 3]
        pose_criterion = self.check_pose_criterion(R, t)
        
        # Projection matrices
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t.reshape(-1, 1)))

        # Triangulate points
        points_4d_homg = cv2.triangulatePoints(P1, P2, pts1.T, pts0.T)
        points_3d = (points_4d_homg / points_4d_homg[3])[:3].T
    
        return points_3d, pose_criterion


class SFM_ddad:
    def __init__(self, translation_threshold) -> None:
        self.translation_threshold = translation_threshold
        self.rotation_threshold = 5.0

    def check_pose_criterion(self, R, t):
        
        # Calculate the magnitude of the translation vector
        translation_magnitude = np.linalg.norm(t)

        # Calculate the rotation angle from the rotation matrix
        rotation_angle = np.arccos((np.trace(R) - 1) / 2) * (180 / np.pi)
        
        return translation_magnitude > self.translation_threshold or rotation_angle > self.rotation_threshold
    
    def __call__(self, data_sample, pts0, pts1):
        K = data_sample['intrinsics']

        # Relative pose (assuming you have R and t)
        R = data_sample['pose_context'][0][:3, :3]
        t = data_sample['pose_context'][0][:3, 3]

        pose_criterion = self.check_pose_criterion(R, t)
        
        # Projection matrices
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t.reshape(-1, 1)))

        # Triangulate points
        points_4d_homg = cv2.triangulatePoints(P1, P2, pts1.T, pts0.T)
        points_3d = (points_4d_homg / points_4d_homg[3])[:3].T
    
        return points_3d, pose_criterion
    

class StereoSGBM:
    def __init__(self) -> None:
        numDisparities = [64, 96, 128, 160] #
        self.stereo_matchers = []
        for blockSize in [1, 2, 3]: #
            for numDisparity in numDisparities:

                sad_window_size = 3
                stereo_params = dict(
                    preFilterCap=63,
                    P1=sad_window_size * sad_window_size * 4,
                    P2=sad_window_size * sad_window_size * 32,
                    minDisparity=0,
                    numDisparities=numDisparity,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=16,
                    blockSize=blockSize)
                stereo_matcher = cv2.StereoSGBM_create(**stereo_params)
                self.stereo_matchers.append(stereo_matcher)
                
    def get_disp(self, im_l, im_r, *args: Any, **kwds: Any) -> Any:
        disps = []
        for stereo in self.stereo_matchers:
            disps.append(stereo.compute(im_l, im_r) / 16)

        disps = np.stack(disps)        
        valids = disps > 0
        disp = ((disps * valids) / (valids.sum(0) + 1e-7)).sum(0)
            
        return disp, valids.max(0)

class Rescaler:
    def __init__(self, config) -> None:
        self.num_beams = config.num_beams
        self.line_size = 2
        
        if config.rescale_with == 'lidar' and self.num_beams is not None:
            self.get_mask = self.get_lidar_mask
        else:
            self.get_mask = lambda x: x

    def get_lidar_mask(self, lidar_map, image_centered=False):
        mask = np.ones_like(lidar_map)
        if image_centered:
            min_height, max_height = 0, lidar_map.shape[0]
        else:
            min_height = np.min(np.where(lidar_map > 0)[0])
            max_height = np.max(np.where(lidar_map > 0)[0])
        diff_height = max_height - min_height
        spacing = diff_height // (self.num_beams + 1)
        current_min = min_height
        for i in range(self.num_beams):
            mask[current_min : current_min + spacing - self.line_size] = False
            current_min += spacing
        mask[current_min:] = False
        
        return mask

    def __call__(self, ref, pred, mask):
        valid_mask = np.logical_and(np.logical_and(mask, ref > 0.01), self.get_mask(mask))
        ref_disp = 1 / ref[valid_mask]
        ransac = RANSACRegressor(LinearRegression(fit_intercept=True)).fit(pred[valid_mask].reshape(-1, 1), ref_disp.reshape(-1, 1))
        disp_rescaled = pred * ransac.estimator_.coef_[0, 0] + ransac.estimator_.intercept_[0]
        
        return 1 / (np.abs(disp_rescaled) + 1e-3)
    
class RescalerPC:
    def __init__(self, config) -> None:
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth
    
    def __call__(self, depth_pts, pts2d, pred, confidence=None):
        depth_pts = depth_pts[:, 2]
        valid_depth = self.min_depth < depth_pts
        pts2d = pts2d[valid_depth]
        ref_disp = 1 / depth_pts[valid_depth]
        pred_match = cv2.remap(pred, pts2d[:, 0], pts2d[:, 1], cv2.INTER_LINEAR)[:, 0]
        ransac = RANSACRegressor(LinearRegression(fit_intercept=True))
        
        model = ransac.fit(pred_match.reshape(-1, 1), ref_disp.reshape(-1, 1), sample_weight=confidence)
        disp_rescaled = pred * model.estimator_.coef_[0, 0] + model.estimator_.intercept_[0]
        depth_rescaled = 1 / (np.abs(disp_rescaled) + 1e-3)

        return depth_rescaled
        