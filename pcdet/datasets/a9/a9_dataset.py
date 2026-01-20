"""
A9 Dataset for OpenPCDet.

A9 is a roadside infrastructure LiDAR dataset with:
- Fixed sensor position (higher viewpoint)
- Single-frame data (no sweeps typically)
- Different class distribution than vehicle-mounted datasets
"""

import copy
import pickle
from pathlib import Path

import numpy as np

from ...utils import box_utils
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

        Binary files should be in (N, 5) format: x, y, z, intensity, timestamp.
        - New preprocessing saves 5 features directly
        - For backward compatibility with 4-feature files, we pad timestamp=0
        """
        lidar_file = Path(self.a9_infos[idx]['point_cloud_path'])
        if not lidar_file.is_absolute():
            lidar_file = self.root_path / lidar_file

        points = np.fromfile(lidar_file, dtype=np.float32)
        total_values = points.size

        # Try 5 features first (new preprocessing format)
        if total_values % 5 == 0:
            points = points.reshape(-1, 5)
        # Fallback to 4 features (legacy KITTI format)
        elif total_values % 4 == 0:
            points = points.reshape(-1, 4)
            # Pad timestamp=0 for roadside sensors
            padding = np.zeros((points.shape[0], 1), dtype=np.float32)
            points = np.hstack([points, padding])
        else:
            # Last resort: truncate to nearest multiple of 5
            valid_count = (total_values // 5) * 5
            logger = getattr(self, 'logger', None)
            if logger:
                logger.warning(f'Irregular point cloud size {total_values}, truncating to {valid_count}')
            points = points[:valid_count].reshape(-1, 5)

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

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Convert prediction to standard format.
        
        Args:
            batch_dict: dict containing frame_id and other batch info
            pred_dicts: list of prediction dicts from model
            class_names: list of class names
            output_path: optional path to save results
        
        Returns:
            list of annotation dicts
        """
        annos = []
        
        for idx, pred_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][idx]
            
            pred_boxes = pred_dict['pred_boxes'].cpu().numpy()
            pred_scores = pred_dict['pred_scores'].cpu().numpy()
            pred_labels = pred_dict['pred_labels'].cpu().numpy()
            
            # Convert labels to class names
            pred_names = np.array([class_names[l - 1] for l in pred_labels])
            
            anno = {
                'frame_id': frame_id,
                'name': pred_names,
                'boxes_lidar': pred_boxes,
                'score': pred_scores,
                'pred_labels': pred_labels,
            }
            annos.append(anno)
            
            # Optionally save to file
            if output_path is not None:
                import pickle
                output_file = output_path / f'{frame_id}.pkl'
                with open(output_file, 'wb') as f:
                    pickle.dump(anno, f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        """
        Evaluate detection results using KITTI-style evaluation.
        
        Tries numba CUDA-based KITTI evaluation first, falls back to simple
        CPU-based statistics if numba CUDA is not available.
        
        Args:
            det_annos: list of detection annotation dicts
            class_names: list of class names
            **kwargs: additional arguments (output_path, etc.)
        
        Returns:
            tuple: (result_str, result_dict)
        """
        if 'annos' not in self.a9_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}
        
        # Try KITTI-style evaluation with numba CUDA
        try:
            return self._kitti_evaluation(det_annos, class_names, **kwargs)
        except Exception as e:
            self.logger.warning(f'KITTI evaluation failed ({e}), using simple evaluation')
            return self._simple_evaluation(det_annos, class_names, **kwargs)
    
    def _kitti_evaluation(self, det_annos, class_names, **kwargs):
        """KITTI-style evaluation using numba CUDA for IoU calculation."""
        from ..kitti.kitti_object_eval_python import eval as kitti_eval
        from ..kitti import kitti_utils
        
        # Map A9 class names to KITTI format for evaluation
        map_name_to_kitti = {
            'car': 'Car',
            'truck': 'Car',
            'bus': 'Car',
            'trailer': 'Car',
            'motorcycle': 'Cyclist',
            'bicycle': 'Cyclist',
            'pedestrian': 'Pedestrian',
        }
        
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.a9_infos]
        
        # Transform annotations to KITTI format
        kitti_utils.transform_annotations_to_kitti_format(
            eval_det_annos, map_name_to_kitti=map_name_to_kitti
        )
        kitti_utils.transform_annotations_to_kitti_format(
            eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
            info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
        )
        
        # Get unique KITTI class names
        kitti_class_names = list(set(map_name_to_kitti.get(x, x) for x in class_names))
        kitti_class_names = [x for x in ['Car', 'Pedestrian', 'Cyclist'] if x in kitti_class_names]
        
        if not kitti_class_names:
            return 'No valid classes for KITTI evaluation', {}
        
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        
        return ap_result_str, ap_dict
    
    def _simple_evaluation(self, det_annos, class_names, **kwargs):
        """Simple CPU-based evaluation (fallback when numba CUDA unavailable)."""
        total_gt = 0
        total_det = 0
        class_stats = {cls: {'gt': 0, 'det': 0} for cls in class_names}
        
        for idx, det_anno in enumerate(det_annos):
            gt_anno = self.a9_infos[idx]['annos']
            
            # Count ground truth
            gt_names = gt_anno.get('name', [])
            if isinstance(gt_names, np.ndarray):
                gt_names = gt_names.tolist()
            for name in gt_names:
                if name in class_stats:
                    class_stats[name]['gt'] += 1
                    total_gt += 1
            
            # Count detections
            det_names = det_anno.get('name', [])
            if isinstance(det_names, np.ndarray):
                det_names = det_names.tolist()
            for name in det_names:
                if name in class_stats:
                    class_stats[name]['det'] += 1
                    total_det += 1
        
        # Build result string
        result_lines = [
            '=' * 60,
            'A9 Evaluation Results (Simple Stats - Fallback Mode)',
            '=' * 60,
            f'Total GT objects: {total_gt}',
            f'Total Detections: {total_det}',
            '-' * 60,
            'Per-class statistics:',
        ]
        
        result_dict = {}
        for cls in class_names:
            gt_count = class_stats[cls]['gt']
            det_count = class_stats[cls]['det']
            result_lines.append(f'  {cls:15s}: GT={gt_count:5d}, Det={det_count:5d}')
            result_dict[f'{cls}/gt_count'] = gt_count
            result_dict[f'{cls}/det_count'] = det_count
        
        result_lines.append('=' * 60)
        
        result_str = '\n'.join(result_lines)
        return result_str, result_dict

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
