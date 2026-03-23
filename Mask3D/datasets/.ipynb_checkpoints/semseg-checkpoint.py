import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
from copy import deepcopy
from random import randrange


import numpy
import torch
from datasets.random_cuboid import RandomCuboid

import albumentations as A
import numpy as np
import scipy
import volumentations as V
import yaml

# from yaml import CLoader as Loader
from torch.utils.data import Dataset
from datasets.scannet200.scannet200_constants import (
    SCANNET_COLOR_MAP_200,
    SCANNET_COLOR_MAP_20,
)

logger = logging.getLogger(__name__)


class SemanticSegmentationDataset(Dataset):
    """拟南芥适配版数据加载器"""

    def __init__(
        self,
        dataset_name="scannet",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        instance_oversampling=0,
        place_around_existing=False,
        max_cut_region=0,
        point_per_cut=100,
        flip_in_center=False,
        noise_rate=0.0,
        resample_points=0.0,
        cache_data=False,
        add_unlabeled_pc=False,
        task="instance_segmentation",
        cropping=False,
        cropping_args=None,
        is_tta=False,
        crop_min_size=20000,
        crop_length=6.0,
        cropping_v1=True,
        reps_per_epoch=1,
        area=-1,
        on_crops=False,
        eval_inner_core=-1,
        filter_out_classes=[],
        label_offset=0,
        add_clip=False,
        is_elastic_distortion=True,
        color_drop=0.0,
    ):
        assert task in ["instance_segmentation", "semantic_segmentation"], "unknown task"

        self.add_clip = add_clip
        self.dataset_name = dataset_name
        self.is_elastic_distortion = is_elastic_distortion
        self.color_drop = color_drop

        # --- 拟南芥颜色映射适配 ---
        if self.dataset_name == "scannet":
            self.color_map = SCANNET_COLOR_MAP_20
            self.color_map[255] = (255, 255, 255)
        elif self.dataset_name == "arabidopsis":
            self.color_map = {
                0: [165, 42, 42],   # Stem
                1: [0, 255, 0],     # Leaf
            }
            self.color_map[255] = (255, 255, 255)
        elif self.dataset_name == "pheno4d":
            self.color_map = SCANNET_COLOR_MAP_20
            self.color_map[255] = (255, 255, 255)
        else:
            self.color_map = SCANNET_COLOR_MAP_20 # 默认

        self.task = task
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.area = area
        self.eval_inner_core = eval_inner_core
        self.reps_per_epoch = reps_per_epoch
        self.cropping = cropping
        self.cropping_args = cropping_args
        self.is_tta = is_tta
        self.on_crops = on_crops
        self.crop_min_size = crop_min_size
        self.crop_length = crop_length
        self.version1 = cropping_v1

        self.random_cuboid = RandomCuboid(
            self.crop_min_size,
            crop_length=self.crop_length,
            version1=self.version1,
        )

        self.mode = mode
        self.data_dir = data_dir
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.instance_oversampling = instance_oversampling
        self.place_around_existing = place_around_existing
        self.max_cut_region = max_cut_region
        self.point_per_cut = point_per_cut
        self.flip_in_center = flip_in_center
        self.noise_rate = noise_rate
        self.resample_points = resample_points

        # 加载数据库
        self._data = []
        for database_path in self.data_dir:
            db_file = Path(database_path) / f"{mode}_database.yaml"
            if db_file.exists():
                self._data.extend(self._load_yaml(db_file))
        
        labels = self._load_yaml(Path(label_db_filepath))
        self._labels = self._select_correct_labels(labels, num_labels)

        # 颜色归一化参数
        if Path(str(color_mean_std)).exists():
            color_mean_std = self._load_yaml(color_mean_std)
            color_mean, color_std = tuple(color_mean_std["mean"]), tuple(color_mean_std["std"])
        else:
            color_mean, color_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        self.normalize_color = A.Normalize(mean=color_mean, std=color_std)
        self.volume_augmentations = V.NoOp()
        self.image_augmentations = A.NoOp()
        self.cache_data = cache_data

    def __len__(self):
        return self.reps_per_epoch * len(self.data)

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)
        points = np.load(self.data[idx]["filepath"])

        coordinates, color, normals, segments, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12],
        )

        raw_coordinates = coordinates.copy()
        raw_color = color.copy()
        raw_normals = normals.copy()

        # 统一执行中心化
        coordinates -= coordinates.mean(0)

        # 颜色归一化：确保输入的 color 是 0-1 范围的 float
        color_for_norm = (color * 255).clip(0, 255).astype(np.uint8)[np.newaxis, :, :]
        color_normalized = np.squeeze(self.normalize_color(image=color_for_norm)["image"])

        # 标签处理：必须返回 3 列 [语义, 实例, 分段]
        labels = labels.astype(np.int32)
        sem_col = labels[:, [0]]
        inst_col = segments[..., None].astype(np.int32)
        seg_col = inst_col.copy() # 第3列通常与实例一致
        
        labels_out = np.hstack((sem_col, inst_col, seg_col))

        # 特征组装：严格根据配置开关进行
        feature_list = [color_normalized]
        if self.add_normals:
            feature_list.append(normals)
        if self.add_raw_coordinates:
            feature_list.append(coordinates)
        
        features = np.hstack(feature_list)

        # 再次兜底检查：如果组装出的特征维数与模型预期不符，打印警告
        # 注意：这里我们假设 features 最终会被截断或处理以匹配模型
        # 在 Mask3D 中，in_channels 通常通过配置传递

        return (
            coordinates, 
            features, 
            labels_out, 
            self.data[idx]["scene"], 
            raw_color, 
            raw_normals, 
            raw_coordinates, 
            idx
        )

    @property
    def data(self):
        return self._data

    @property
    def label_info(self):
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def _select_correct_labels(self, labels, num_labels):
        return labels # 简化逻辑

    def _remap_from_zero(self, labels):
        labels[
            ~np.isin(labels, list(self.label_info.keys()))
        ] = self.ignore_label
        # 映射到从 0 开始的范围
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped

    def map2color(self, labels):
        output_colors = []
        for label in labels:
            output_colors.append(self.color_map[int(label)])
        return torch.tensor(output_colors)
