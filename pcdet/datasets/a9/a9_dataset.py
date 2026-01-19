"""
A9 Dataset for OpenPCDet.

A9 is a roadside infrastructure LiDAR dataset with:
- Fixed sensor position (higher viewpoint)
- Single-frame data (no sweeps typically)
- Different class distribution than vehicle-mounted datasets
"""

import copy
import pickle
import os
from pathlib import Path

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class A9Dataset(DatasetTemplate):
    """A9 Roadside LiDAR Dataset for OpenPCDet."""

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            dataset_cfg: Configuration for the dataset
            class_names: List of class names to detect
            training: Whether in training mode
            root_path: Root path to the dataset
            logger: Logger instance
        """
        # Use project root for relative paths
        from ...config import cfg
        data_path = dataset_cfg.DATA_PATH if root_path is None else root_path
        if not str(data_path).startswith('/') and not str(data_path).startswith('C:') and ':' not in str(data_path):
            # Relative path - resolve from project root
            effective_root = cfg.PROJECT_ROOT / data_path
        else:
            effective_root = root_path

        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=effective_root, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT.get(self.mode, 'train')

        # A9 uses pickle infos directly, no ImageSets needed
        self.a9_infos = []
        self.include_data(self.mode)

    def include_data(self, mode):
        """Load dataset info from pickle files."""
        self.logger.info(f'Loading A9 dataset for mode: {mode}')
        a9_infos = []

        for info_path in self.dataset_cfg.INFO_PATH.get(mode, []):
            info_path = Path(info_path)
            # Resolve relative paths from root_path
            if not info_path.is_absolute():
                info_path = self.root_path / info_path

            if not info_path.exists():
                self.logger.warning(f'Info file not found: {info_path}')
                continue

            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                a9_infos.extend(infos)

        self.a9_infos.extend(a9_infos)
        self.logger.info(f'Total samples for A9 dataset ({mode}): {len(a9_infos)}')

    def get_lidar(self, idx):
        """Load point cloud data from binary file.

        Handles variable point cloud formats by trying different reshape options.
        Binary files are expected to be in KITTI format: (N, 4) with x, y, z, intensity.
        For OpenPCDet with timestamp feature, we pad to 5 features (x, y, z, intensity, timestamp).
        """
        lidar_file = Path(self.a9_infos[idx]['point_cloud_path'])
        if not lidar_file.is_absolute():
            lidar_file = self.root_path / lidar_file

        with open(lidar_file, 'rb') as f:
            points = np.fromfile(f, dtype=np.float32)

        # Try to reshape to common formats
        total_points = points.size

        # Try common formats (4 for KITTI, 5 for OpenPCDet with timestamp)
        for num_features in [4, 5, 3, 6]:
            if total_points % num_features == 0:
                points = points.reshape(-1, num_features)
                logger = getattr(self, 'logger', None)
                if logger:
                    logger.debug(f'Loaded point cloud with {total_points // num_features} points, {num_features} features')
                break
        else:
            # If no standard format works, truncate to nearest divisible by 5
            # and pad if needed to ensure at least 5 features
            valid_points = (total_points // 5) * 5
            points = points[:valid_points]
            points = points.reshape(-1, 5)
            logger = getattr(self, 'logger', None)
            if logger:
                logger.warning(f'Point cloud file has irregular size {total_points}. '
                               f'Truncated to {valid_points} values and reshaped to (N, 5)')

        # Ensure we have exactly 5 features (x, y, z, intensity, timestamp) for OpenPCDet compatibility
        # OpenPCDet expects 5 features by default for A9 dataset (with timestamp)
        if points.shape[1] > 5:
            points = points[:, :5]
        elif points.shape[1] < 5:
            # Pad with zeros if less than 5 features
            padding = np.zeros((points.shape[0], 5 - points.shape[1]), dtype=np.float32)
            points = np.hstack([points, padding])

        return points

    def get_label(self, idx):
        """Load label data (boxes and names) from info dict."""
        annos = self.a9_infos[idx]['annos']
        gt_boxes = annos.get('gt_boxes_lidar', np.zeros((0, 9), dtype=np.float32))
        gt_names = annos.get('name', [])
        return gt_boxes, gt_names

    def get_annos(self, idx):
        """Get annotations in OpenPCDet format."""
        info = self.a9_infos[idx]
        annos = info.get('annos', {})
        return {
            'boxes_lidar': annos.get('gt_boxes_lidar', np.zeros((0, 9), dtype=np.float32)),
            'names': annos.get('name', []),
            'scores': annos.get('score', []),
            'labels': annos.get('obj_labels', []),
        }

    @staticmethod
    def get_fov_points(gt_boxes, points, num_point_features=4):
        """Filter points outside the field of view."""
        if gt_boxes.shape[0] == 0:
            return points
        box_utils.remove_points_in_boxes(points, gt_boxes)
        return points

    def generate_prediction_dicts(self, batch_dict, pred_dicts):
        """Convert prediction to standard format."""
        pred_dicts = batch_dict['pred_dicts']
        annos = []

        for i, pred_dict in enumerate(pred_dicts):
            pred_boxes = pred_dict['pred_boxes']
            pred_scores = pred_dict['pred_scores']
            pred_labels = pred_dict['pred_labels']

            anno = {
                'boxes_lidar': pred_boxes.cpu().numpy(),
                'score': pred_scores.cpu().numpy(),
                'label': pred_labels.cpu().numpy(),
            }
            annos.append(anno)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        """Evaluate detection results (placeholder for custom metrics)."""
        # For now, use basic KITTI-style evaluation
        from ...utils import eval_utils
        ap_result_str, ap_dict = eval_utils.get_coco_eval_result(
            det_annos, class_names, current_classes=class_names
        )
        return ap_result_str, ap_dict

    def set_split(self, split):
        """Set the current data split."""
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.a9_infos = []
        self.include_data(self.mode)

    def __len__(self):
        return len(self.a9_infos)

    def __getitem__(self, index):
        """Get single sample."""
        info = copy.deepcopy(self.a9_infos[index])

        points = self.get_lidar(index)
        gt_boxes, gt_names = self.get_label(index)

        input_dict = {
            'points': points,
            'frame_id': info.get('token', str(index)),
            'gt_names': gt_names,
            'gt_boxes': gt_boxes,
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
