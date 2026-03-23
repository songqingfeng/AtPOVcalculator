import gc
from contextlib import nullcontext
from pathlib import Path
import statistics
import shutil
import os
import math
import traceback
import pyviz3d.visualizer as vis
from torch_scatter import scatter_mean
import matplotlib
try:
    from evaluate_semantic_instance_nnj import evaluate
except ImportError:
    try:
        from evaluate_semantic_instance import evaluate
    except ImportError:
        from benchmark.evaluate_semantic_instance import evaluate
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from utils.votenet_utils.eval_det import eval_det
from datasets.scannet200.scannet200_splits import (
    HEAD_CATS_SCANNET_200,
    TAIL_CATS_SCANNET_200,
    COMMON_CATS_SCANNET_200,
    VALID_CLASS_IDS_200_VALIDATION,
)

import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from models.metrics import IoU
import random
import colorsys
from typing import List, Tuple
import functools


VISUALIZATION_MINIMAL_MODE = True


@functools.lru_cache(20)
def get_evenly_distributed_colors(
    count: int,
) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(
        map(
            lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(
                np.uint8
            ),
            HSV_tuples,
        )
    )


def write_rgb_ply(points, colors, filename):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors)

    if points.shape[0] == 0:
        return

    if colors.dtype != np.uint8:
        if colors.max() <= 1.0:
            colors = (colors * 255).clip(0, 255)
        colors = colors.astype(np.uint8)

    with open(filename, "w", encoding="ascii") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {points.shape[0]}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        for point, color in zip(points, colors):
            ply_file.write(
                f"{point[0]} {point[1]} {point[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def to_numpy_array(data):
    if isinstance(data, np.ndarray):
        return data
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def write_instance_semantic_ply(
    points,
    semantic_ids,
    instance_ids,
    filename,
    point_indices=None,
    colors=None,
):
    points = np.asarray(points, dtype=np.float32)
    semantic_ids = np.asarray(semantic_ids, dtype=np.int32).reshape(-1)
    instance_ids = np.asarray(instance_ids, dtype=np.int32).reshape(-1)

    if points.shape[0] == 0:
        return

    num_points = points.shape[0]
    if semantic_ids.shape[0] != num_points or instance_ids.shape[0] != num_points:
        raise ValueError("semantic_ids/instance_ids size must match points size")

    if point_indices is None:
        point_indices = np.arange(num_points, dtype=np.int32)
    else:
        point_indices = np.asarray(point_indices, dtype=np.int32).reshape(-1)
        if point_indices.shape[0] != num_points:
            raise ValueError("point_indices size must match points size")

    if colors is None:
        colors = np.zeros((num_points, 3), dtype=np.uint8)
    else:
        colors = np.asarray(colors)
        if colors.dtype != np.uint8:
            if colors.max() <= 1.0:
                colors = (colors * 255).clip(0, 255)
            colors = colors.astype(np.uint8)
        if colors.shape[0] != num_points:
            raise ValueError("colors size must match points size")

    with open(filename, "w", encoding="ascii") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {num_points}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("property int semantic_id\n")
        ply_file.write("property int instance_id\n")
        ply_file.write("property int point_index\n")
        ply_file.write("end_header\n")

        for i in range(num_points):
            point = points[i]
            color = colors[i]
            ply_file.write(
                f"{point[0]} {point[1]} {point[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{int(semantic_ids[i])} {int(instance_ids[i])} {int(point_indices[i])}\n"
            )


class RegularCheckpointing(pl.Callback):
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        general = pl_module.config.general
        trainer.save_checkpoint(f"{general.save_dir}/last-epoch.ckpt")
        print("Checkpoint created")


class InstanceSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.decoder_id = config.general.decoder_id

        if config.model.train_on_segments:
            self.mask_type = "segment_mask"
        else:
            self.mask_type = "masks"

        self.eval_on_segments = config.general.eval_on_segments

        self.config = config
        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext
        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad
        # loss
        self.ignore_label = config.data.ignore_label

        matcher = hydra.utils.instantiate(config.matcher)
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
            "loss_iou": getattr(self.config.general, "iou_head_loss_weight", 0.0),
        }

        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            if i not in self.config.general.ignore_mask_idx:
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()}
                )
            else:
                aux_weight_dict.update(
                    {k + f"_{i}": 0.0 for k, v in weight_dict.items()}
                )
        weight_dict.update(aux_weight_dict)

        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()

        self.criterion = hydra.utils.instantiate(
            config.loss, matcher=matcher, weight_dict=weight_dict
        )

        # metrics
        self.confusion = hydra.utils.instantiate(config.metrics)
        self.iou = IoU()
        # misc
        self.labels_info = dict()

    def forward(
        self, x, point2segment=None, raw_coordinates=None, is_eval=False
    ):
        with self.optional_freeze():
            x = self.model(
                x,
                point2segment,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
            )
        return x

    def training_step(self, batch, batch_idx):
        data, target, file_names = batch

        if data.features.shape[0] > self.config.general.max_batch_size:
            print("data exceeds threshold")
            raise RuntimeError("BATCH TOO BIG")

        if len(target) == 0:
            print("no targets")
            return None

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]
        else:
            raw_coordinates = data.coordinates[:, 1:].float() * self.model.voxel_size

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        try:
            output = self.forward(
                data,
                point2segment=[
                    target[i]["point2segment"] for i in range(len(target))
                ],
                raw_coordinates=raw_coordinates,
            )
        except RuntimeError as run_err:
            print(run_err)
            if (
                "only a single point gives nans in cross-attention"
                == run_err.args[0]
            ):
                return None
            else:
                raise run_err

        try:
            losses = self.criterion(output, target, mask_type=self.mask_type)
            if getattr(self.config.general, "iou_head_loss_weight", 0.0) > 0 and ("pred_iou" in output):
                idx_cached = getattr(self.criterion, "last_indices", None)
                indices = idx_cached if idx_cached is not None else self.criterion.matcher(
                    {"pred_logits": output["pred_logits"], "pred_masks": output["pred_masks"]},
                    target,
                    self.mask_type,
                )
                iou_losses = []
                max_pairs = int(getattr(self.config.general, "iou_max_pairs", 0) or 0)
                point_ratio = float(getattr(self.config.general, "iou_point_ratio", 0.0) or 0.0)
                for b, (map_id, target_id) in enumerate(indices):
                    if len(map_id) == 0:
                        continue
                    if max_pairs > 0 and len(map_id) > max_pairs:
                        map_id = map_id[:max_pairs]
                        target_id = target_id[:max_pairs]
                    pred = output["pred_masks"][b][:, map_id].T
                    tgt = target[b][self.mask_type][target_id].float()
                    if point_ratio > 0.0:
                        num = max(1, int(point_ratio * tgt.shape[1]))
                        point_idx = torch.randperm(tgt.shape[1], device=tgt.device)[:num]
                        pred = pred[:, point_idx]
                        tgt = tgt[:, point_idx]
                    elif self.criterion.num_points != -1:
                        num = int(self.criterion.num_points * tgt.shape[1])
                        point_idx = torch.randperm(tgt.shape[1], device=tgt.device)[:num]
                        pred = pred[:, point_idx]
                        tgt = tgt[:, point_idx]
                    pred = pred.sigmoid()
                    inter = (pred * tgt).sum(-1)
                    union = pred.sum(-1) + tgt.sum(-1) - inter
                    soft_iou = (inter + 1.0) / (union + 1.0)
                    pred_iou = output["pred_iou"][b][map_id]
                    iou_losses.append(torch.nn.functional.smooth_l1_loss(pred_iou, soft_iou, reduction="mean"))
                if len(iou_losses) > 0:
                    losses["loss_iou"] = torch.stack(iou_losses).mean()
        except ValueError as val_err:
            print(f"ValueError: {val_err}")
            print(f"data shape: {data.shape}")
            print(f"data feat shape:  {data.features.shape}")
            print(f"data feat nans:   {data.features.isnan().sum()}")
            print(f"output: {output}")
            print(f"target: {target}")
            print(f"filenames: {file_names}")
            raise val_err

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        logs = {
            f"train_{k}": v.detach().cpu().item() for k, v in losses.items()
        }

        logs["train_mean_loss_ce"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_ce" in k]]
        )

        logs["train_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_mask" in k]]
        )

        logs["train_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_dice" in k]]
        )

        self.log_dict(logs)
        return sum(losses.values())

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def export(self, pred_masks, scores, pred_classes, file_names, decoder_id):
        root_path = f"eval_output"
        base_path = f"{root_path}/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}/decoder_{decoder_id}"
        pred_mask_path = f"{base_path}/pred_mask"
        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        orig_name = file_names
        safe_name = str(Path(orig_name).with_suffix(""))
        out_txt_path = Path(base_path) / f"{safe_name}.txt"
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_txt_path, "w") as fout:
            real_id = -1
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                if score > self.config.general.export_threshold:
                    mask_out_path = Path(pred_mask_path) / f"{safe_name}_{real_id}.txt"
                    mask_out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savetxt(mask_out_path, mask, fmt="%d")
                    fout.write(
                        f"pred_mask/{safe_name}_{real_id}.txt {pred_class} {score}\n"
                    )

    def training_epoch_end(self, outputs):
        train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(
            outputs
        )
        results = {"train_loss_mean": train_loss}
        self.log_dict(results)

    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)

    def _get_visualization_postprocess_params(self):
        voxel_size = float(getattr(self.config.data, "voxel_size", 1.0))
        component_eps = float(
            getattr(
                self.config.general,
                "visualization_component_eps",
                max(voxel_size * 2.5, 1e-4),
            )
        )
        min_component_points = int(
            getattr(
                self.config.general,
                "visualization_min_component_points",
                10,
            )
        )
        keep_largest_component = bool(
            getattr(
                self.config.general,
                "visualization_keep_largest_component",
                False,
            )
        )
        return component_eps, max(1, min_component_points), keep_largest_component

    def _get_visualization_instance_point_threshold(self):
        min_component_points = int(
            getattr(
                self.config.general,
                "visualization_min_component_points",
                10,
            )
        )
        return max(
            min_component_points,
            int(
                getattr(
                    self.config.general,
                    "visualization_min_final_instance_points",
                    30,
                )
            ),
        )

    def _get_visualization_reassignment_distance(self):
        voxel_size = float(getattr(self.config.data, "voxel_size", 1.0))
        return float(
            getattr(
                self.config.general,
                "visualization_reassign_distance",
                max(voxel_size * 6.0, 1e-4),
            )
        )

    def _get_visualization_reassignment_heatmap_threshold(self):
        return float(
            getattr(
                self.config.general,
                "visualization_reassign_heatmap_threshold",
                0.35,
            )
        )

    def _get_visualization_split_component_ratio(self):
        return float(
            getattr(
                self.config.general,
                "visualization_split_component_ratio",
                0.2,
            )
        )

    def _extract_connected_components(
        self,
        coords_raw,
        mask_bool,
        eps,
        min_component_points,
        keep_largest_component=False,
    ):
        mask_bool = np.asarray(mask_bool, dtype=bool).reshape(-1)
        point_ids = np.flatnonzero(mask_bool)
        if point_ids.size == 0 or point_ids.size < min_component_points:
            return []

        if point_ids.size == 1 or eps <= 0.0:
            component_mask = np.zeros(mask_bool.shape[0], dtype=bool)
            component_mask[point_ids] = True
            return [component_mask]

        try:
            cluster_ids = DBSCAN(
                eps=eps,
                min_samples=1,
                n_jobs=-1,
            ).fit(np.asarray(coords_raw)[point_ids]).labels_
        except Exception:
            component_mask = np.zeros(mask_bool.shape[0], dtype=bool)
            component_mask[point_ids] = True
            return [component_mask]

        cluster_labels, cluster_sizes = np.unique(cluster_ids, return_counts=True)
        if keep_largest_component:
            largest_cluster = cluster_labels[np.argmax(cluster_sizes)]
            cluster_labels = np.asarray([largest_cluster])

        components = []
        for cluster_label in cluster_labels:
            keep_ids = point_ids[cluster_ids == cluster_label]
            if keep_ids.size < min_component_points:
                continue
            component_mask = np.zeros(mask_bool.shape[0], dtype=bool)
            component_mask[keep_ids] = True
            components.append(component_mask)
        return components

    def _filter_connected_components(
        self,
        coords_raw,
        mask_bool,
        eps,
        min_component_points,
        keep_largest_component=True,
    ):
        mask_bool = np.asarray(mask_bool, dtype=bool).reshape(-1)
        component_mask = np.zeros(mask_bool.shape[0], dtype=bool)
        components = self._extract_connected_components(
            coords_raw,
            mask_bool,
            eps,
            min_component_points,
            keep_largest_component=keep_largest_component,
        )
        for component in components:
            component_mask |= component
        return component_mask

    def _assign_points_to_nearest_instances(
        self,
        coords_raw,
        pred_semantic_ids,
        pred_instance_ids,
        orphan_mask,
        max_distance=0.0,
        preferred_semantic_ids=None,
        same_label_only=False,
    ):
        orphan_mask = np.asarray(orphan_mask, dtype=bool).reshape(-1)
        if not np.any(orphan_mask):
            return 0

        assigned_mask = pred_instance_ids > 0
        if not np.any(assigned_mask):
            return 0

        coords_raw = np.asarray(coords_raw, dtype=np.float32)
        assigned_semantic_ids = pred_semantic_ids[assigned_mask]
        assigned_instance_ids = pred_instance_ids[assigned_mask]
        assigned_coords = coords_raw[assigned_mask]

        if same_label_only:
            if preferred_semantic_ids is None:
                return 0
            preferred_semantic_ids = np.asarray(
                preferred_semantic_ids, dtype=np.int32
            ).reshape(-1)
            assigned_total = 0
            for label_value in np.unique(preferred_semantic_ids[orphan_mask]):
                if label_value < 0:
                    continue

                orphan_label_mask = orphan_mask & (
                    preferred_semantic_ids == label_value
                )
                if not np.any(orphan_label_mask):
                    continue

                assigned_label_ids = np.flatnonzero(
                    assigned_semantic_ids == label_value
                )
                if assigned_label_ids.size == 0:
                    continue

                tree = KDTree(assigned_coords[assigned_label_ids])
                orphan_ids = np.flatnonzero(orphan_label_mask)
                distances, nearest_indices = tree.query(
                    coords_raw[orphan_ids],
                    k=1,
                    return_distance=True,
                )
                distances = distances.reshape(-1)
                nearest_indices = nearest_indices.reshape(-1)
                keep_mask = (
                    distances <= max_distance
                    if max_distance > 0
                    else np.ones(orphan_ids.shape[0], dtype=bool)
                )
                if not np.any(keep_mask):
                    continue

                matched_ids = orphan_ids[keep_mask]
                matched_assigned_ids = assigned_label_ids[
                    nearest_indices[keep_mask]
                ]
                pred_semantic_ids[matched_ids] = assigned_semantic_ids[
                    matched_assigned_ids
                ]
                pred_instance_ids[matched_ids] = assigned_instance_ids[
                    matched_assigned_ids
                ]
                assigned_total += int(matched_ids.size)
            return assigned_total

        tree = KDTree(assigned_coords)
        orphan_ids = np.flatnonzero(orphan_mask)
        distances, nearest_indices = tree.query(
            coords_raw[orphan_ids],
            k=1,
            return_distance=True,
        )
        distances = distances.reshape(-1)
        nearest_indices = nearest_indices.reshape(-1)
        keep_mask = (
            distances <= max_distance
            if max_distance > 0
            else np.ones(orphan_ids.shape[0], dtype=bool)
        )
        if not np.any(keep_mask):
            return 0

        matched_ids = orphan_ids[keep_mask]
        matched_assigned_ids = nearest_indices[keep_mask]
        pred_semantic_ids[matched_ids] = assigned_semantic_ids[
            matched_assigned_ids
        ]
        pred_instance_ids[matched_ids] = assigned_instance_ids[
            matched_assigned_ids
        ]
        return int(matched_ids.size)

    def _build_unique_prediction_visualization(
        self,
        full_res_coords_raw,
        full_res_coords_vis,
        original_normals_np,
        sorted_masks,
        sort_classes,
        sort_scores_values,
        sorted_heatmaps,
    ):
        n_points = full_res_coords_raw.shape[0]
        valid_sem_classes = int(self.config.data.num_labels)
        (
            component_eps,
            min_component_points,
            keep_largest_component,
        ) = self._get_visualization_postprocess_params()
        min_final_instance_points = (
            self._get_visualization_instance_point_threshold()
        )
        reassign_distance = self._get_visualization_reassignment_distance()
        reassign_heatmap_threshold = (
            self._get_visualization_reassignment_heatmap_threshold()
        )
        split_component_ratio = (
            self._get_visualization_split_component_ratio()
        )

        best_point_scores = np.full(n_points, -np.inf, dtype=np.float32)
        best_candidate_ids = np.full(n_points, -1, dtype=np.int32)
        fallback_point_scores = np.full(n_points, -np.inf, dtype=np.float32)
        fallback_candidate_ids = np.full(n_points, -1, dtype=np.int32)
        candidate_labels = []
        candidate_scores = []

        for did in range(len(sorted_masks)):
            masks_d = to_numpy_array(sorted_masks[did])
            if masks_d.ndim != 2 or masks_d.shape[1] == 0:
                continue

            classes_d = to_numpy_array(sort_classes[did]).reshape(-1)
            if classes_d.shape[0] != masks_d.shape[1]:
                continue

            scores_d = None
            if sort_scores_values is not None and len(sort_scores_values) > did:
                scores_d = to_numpy_array(sort_scores_values[did]).reshape(-1)
                if scores_d.shape[0] != masks_d.shape[1]:
                    scores_d = None

            heatmaps_d = None
            if sorted_heatmaps is not None and len(sorted_heatmaps) > did:
                heatmaps_d = to_numpy_array(sorted_heatmaps[did])
                if heatmaps_d.shape != masks_d.shape:
                    heatmaps_d = None

            visit_order = (
                np.argsort(scores_d)[::-1]
                if scores_d is not None
                else np.arange(masks_d.shape[1])
            )

            for instance_idx in visit_order:
                label_value = int(classes_d[instance_idx])
                if label_value < 0 or label_value >= valid_sem_classes:
                    continue

                raw_mask = masks_d[:, instance_idx].astype(bool)
                if not np.any(raw_mask):
                    continue

                instance_score = (
                    float(scores_d[instance_idx])
                    if scores_d is not None
                    else 1.0
                )
                heatmap_col = None
                if heatmaps_d is not None:
                    heatmap_col = np.asarray(
                        heatmaps_d[:, instance_idx], dtype=np.float32
                    )

                fallback_mask = raw_mask.copy()
                if heatmap_col is not None:
                    fallback_mask |= (
                        heatmap_col >= reassign_heatmap_threshold
                    )

                min_split_component_points = max(
                    min_component_points,
                    min_final_instance_points,
                    int(np.ceil(raw_mask.sum() * split_component_ratio)),
                )
                candidate_components = self._extract_connected_components(
                    full_res_coords_raw,
                    raw_mask,
                    component_eps,
                    min_split_component_points,
                    keep_largest_component=keep_largest_component,
                )
                if len(candidate_components) == 0:
                    candidate_components = self._extract_connected_components(
                        full_res_coords_raw,
                        raw_mask,
                        component_eps,
                        1,
                        keep_largest_component=keep_largest_component,
                    )
                if len(candidate_components) == 0:
                    continue

                for component_mask in candidate_components:
                    candidate_id = len(candidate_labels)
                    if heatmap_col is not None and np.any(component_mask):
                        component_score = float(
                            instance_score
                            * np.mean(heatmap_col[component_mask])
                        )
                    else:
                        component_score = float(instance_score)

                    candidate_labels.append(label_value)
                    candidate_scores.append(component_score)

                    raw_point_scores = np.full(
                        n_points, -np.inf, dtype=np.float32
                    )
                    if heatmap_col is not None:
                        raw_point_scores[fallback_mask] = (
                            component_score * heatmap_col[fallback_mask]
                        )
                    else:
                        raw_point_scores[fallback_mask] = component_score

                    better_raw_mask = fallback_mask & (
                        raw_point_scores > fallback_point_scores
                    )
                    if np.any(better_raw_mask):
                        fallback_point_scores[better_raw_mask] = (
                            raw_point_scores[better_raw_mask]
                        )
                        fallback_candidate_ids[better_raw_mask] = candidate_id

                    point_scores = np.full(
                        n_points, -np.inf, dtype=np.float32
                    )
                    if heatmap_col is not None:
                        point_scores[component_mask] = (
                            component_score * heatmap_col[component_mask]
                        )
                    else:
                        point_scores[component_mask] = component_score

                    better_mask = component_mask & (
                        point_scores > best_point_scores
                    )
                    if not np.any(better_mask):
                        continue

                    best_point_scores[better_mask] = point_scores[better_mask]
                    best_candidate_ids[better_mask] = candidate_id

        pred_semantic_ids = np.full(n_points, -1, dtype=np.int32)
        pred_instance_ids = np.zeros(n_points, dtype=np.int32)
        final_instance_labels = []
        final_instance_scores = []
        final_instance_point_ids = []

        surviving_candidate_ids = np.unique(
            best_candidate_ids[best_candidate_ids >= 0]
        )
        surviving_candidate_ids = sorted(
            surviving_candidate_ids.tolist(),
            key=lambda cid: candidate_scores[cid],
            reverse=True,
        )

        for candidate_id in surviving_candidate_ids:
            owned_points = best_candidate_ids == candidate_id
            owned_components = self._extract_connected_components(
                full_res_coords_raw,
                owned_points,
                component_eps,
                min_component_points,
                keep_largest_component=False,
            )
            for component_mask in owned_components:
                component_point_ids = np.flatnonzero(component_mask)
                if component_point_ids.size < min_final_instance_points:
                    continue
                final_instance_point_ids.append(component_point_ids)
                final_instance_labels.append(candidate_labels[candidate_id])
                final_instance_scores.append(candidate_scores[candidate_id])

        if len(final_instance_point_ids) == 0 and len(candidate_labels) > 0:
            for candidate_id in surviving_candidate_ids:
                owned_points = best_candidate_ids == candidate_id
                if not np.any(owned_points):
                    continue
                owned_components = self._extract_connected_components(
                    full_res_coords_raw,
                    owned_points,
                    component_eps,
                    1,
                    keep_largest_component=False,
                )
                for component_mask in owned_components:
                    component_point_ids = np.flatnonzero(component_mask)
                    if component_point_ids.size == 0:
                        continue
                    final_instance_point_ids.append(component_point_ids)
                    final_instance_labels.append(candidate_labels[candidate_id])
                    final_instance_scores.append(candidate_scores[candidate_id])

        for new_instance_id, (point_ids, label_value) in enumerate(
            zip(final_instance_point_ids, final_instance_labels),
            start=1,
        ):
            pred_semantic_ids[point_ids] = int(label_value)
            pred_instance_ids[point_ids] = new_instance_id

        fallback_semantic_ids = np.full(n_points, -1, dtype=np.int32)
        valid_fallback_mask = fallback_candidate_ids >= 0
        if np.any(valid_fallback_mask) and len(candidate_labels) > 0:
            candidate_labels_arr = np.asarray(candidate_labels, dtype=np.int32)
            fallback_semantic_ids[valid_fallback_mask] = candidate_labels_arr[
                fallback_candidate_ids[valid_fallback_mask]
            ]

            self._assign_points_to_nearest_instances(
                full_res_coords_raw,
                pred_semantic_ids,
                pred_instance_ids,
                orphan_mask=(pred_instance_ids == 0) & valid_fallback_mask,
                max_distance=reassign_distance,
                preferred_semantic_ids=fallback_semantic_ids,
                same_label_only=True,
            )

        fill_all_unassigned = bool(
            getattr(
                self.config.general,
                "visualization_fill_unassigned_with_nearest",
                True,
            )
        )
        orphan_mask = pred_instance_ids == 0
        if not fill_all_unassigned:
            orphan_mask &= valid_fallback_mask
        self._assign_points_to_nearest_instances(
            full_res_coords_raw,
            pred_semantic_ids,
            pred_instance_ids,
            orphan_mask=orphan_mask,
            max_distance=0.0,
            same_label_only=False,
        )

        num_instances = int(pred_instance_ids.max())
        if num_instances > 0:
            final_masks = np.zeros((n_points, num_instances), dtype=np.uint8)
            final_classes = np.zeros(num_instances, dtype=np.int32)
            final_scores = np.zeros(num_instances, dtype=np.float32)
            for instance_id in range(1, num_instances + 1):
                instance_mask = pred_instance_ids == instance_id
                final_masks[:, instance_id - 1] = instance_mask.astype(np.uint8)
                final_classes[instance_id - 1] = int(
                    pred_semantic_ids[instance_mask][0]
                )
                final_scores[instance_id - 1] = float(
                    np.max(best_point_scores[instance_mask])
                )
        else:
            final_masks = np.zeros((n_points, 0), dtype=np.uint8)
            final_classes = np.zeros((0,), dtype=np.int32)
            final_scores = np.zeros((0,), dtype=np.float32)

        assigned_mask = pred_instance_ids > 0
        if not np.any(assigned_mask):
            empty_uint8 = np.zeros((0, 3), dtype=np.uint8)
            return {
                "pred_masks": final_masks,
                "pred_scores": final_scores,
                "pred_classes": final_classes,
                "pred_semantic_ids": pred_semantic_ids,
                "pred_instance_ids": pred_instance_ids,
                "assigned_mask": assigned_mask,
                "coords_vis": np.zeros((0, 3), dtype=np.float32),
                "coords_raw": np.zeros((0, 3), dtype=np.float32),
                "normals": np.zeros((0, 3), dtype=np.float32),
                "sem_colors": empty_uint8,
                "inst_colors": empty_uint8,
            }

        assigned_coords_vis = full_res_coords_vis[assigned_mask]
        assigned_coords_raw = full_res_coords_raw[assigned_mask]
        assigned_normals = original_normals_np[assigned_mask]
        sem_colors = to_numpy_array(
            self.validation_dataset.map2color(pred_semantic_ids[assigned_mask])
        ).astype(np.uint8)

        instance_palette = np.vstack(
            get_evenly_distributed_colors(max(1, num_instances))
        ).astype(np.uint8)
        inst_colors = instance_palette[pred_instance_ids[assigned_mask] - 1]

        return {
            "pred_masks": final_masks,
            "pred_scores": final_scores,
            "pred_classes": final_classes,
            "pred_semantic_ids": pred_semantic_ids,
            "pred_instance_ids": pred_instance_ids,
            "assigned_mask": assigned_mask,
            "coords_vis": assigned_coords_vis,
            "coords_raw": assigned_coords_raw,
            "normals": assigned_normals,
            "sem_colors": sem_colors,
            "inst_colors": inst_colors,
        }

    def save_visualizations(
        self,
        target_full,
        full_res_coords,
        sorted_masks,
        sort_classes,
        file_name,
        original_colors,
        original_normals,
        sort_scores_values,
        point_size=20,
        sorted_heatmaps=None,
        query_pos=None,
        backbone_features=None,
        point_indices=None,
        pred_vis=None,
    ):
        full_res_coords_raw = np.asarray(full_res_coords, dtype=np.float32).copy()
        full_res_coords_vis = full_res_coords_raw.copy()
        full_res_coords_vis -= full_res_coords_vis.mean(axis=0)
        original_colors_np = np.asarray(original_colors)
        original_normals_np = np.asarray(original_normals)

        if point_indices is None:
            point_indices = np.arange(full_res_coords_raw.shape[0], dtype=np.int32)
        else:
            point_indices = np.asarray(point_indices, dtype=np.int32).reshape(-1)
            if point_indices.shape[0] != full_res_coords_raw.shape[0]:
                raise ValueError("point_indices size must match point cloud size")

        visualization_minimal_mode = bool(
            getattr(
                self.config.general,
                "visualization_minimal_mode",
                VISUALIZATION_MINIMAL_MODE,
            )
        )

        n_points = full_res_coords_raw.shape[0]

        gt_pcd_pos_vis = []
        gt_pcd_pos_raw = []
        gt_pcd_normals = []
        gt_pcd_color = []
        gt_inst_pcd_color = []
        gt_boxes = []
        gt_semantic_ids = np.full(n_points, 255, dtype=np.int32)
        gt_instance_ids = np.zeros(n_points, dtype=np.int32)
        gt_instance_counter = 1

        if "labels" in target_full:
            instances_colors = torch.from_numpy(
                np.vstack(
                    get_evenly_distributed_colors(
                        target_full["labels"].shape[0]
                    )
                )
            )
            for instance_counter, (label, mask) in enumerate(
                zip(target_full["labels"], target_full["masks"])
            ):
                label_value = int(label.item()) if torch.is_tensor(label) else int(label)
                if label_value == 255:
                    continue

                mask_tmp = to_numpy_array(mask).astype(bool)
                mask_coords_vis = full_res_coords_vis[mask_tmp, :]
                mask_coords_raw = full_res_coords_raw[mask_tmp, :]

                if len(mask_coords_vis) == 0:
                    continue

                gt_pcd_pos_vis.append(mask_coords_vis)
                gt_pcd_pos_raw.append(mask_coords_raw)
                mask_coords_min = mask_coords_vis.min(axis=0)
                mask_coords_max = mask_coords_vis.max(axis=0)
                size = mask_coords_max - mask_coords_min
                mask_coords_middle = mask_coords_min + size / 2

                gt_boxes.append(
                    {
                        "position": mask_coords_middle,
                        "size": size,
                        "color": self.validation_dataset.map2color([label_value])[0],
                    }
                )

                gt_pcd_color.append(
                    self.validation_dataset.map2color([label_value]).repeat(
                        mask_coords_vis.shape[0], 1
                    )
                )
                gt_inst_pcd_color.append(
                    instances_colors[instance_counter % len(instances_colors)]
                    .unsqueeze(0)
                    .repeat(mask_coords_vis.shape[0], 1)
                )

                gt_pcd_normals.append(original_normals_np[mask_tmp, :])
                gt_semantic_ids[mask_tmp] = label_value
                gt_instance_ids[mask_tmp] = gt_instance_counter
                gt_instance_counter += 1

            if len(gt_pcd_pos_vis) > 0:
                gt_pcd_pos_vis = np.concatenate(gt_pcd_pos_vis)
                gt_pcd_pos_raw = np.concatenate(gt_pcd_pos_raw)
                gt_pcd_normals = np.concatenate(gt_pcd_normals)
                gt_pcd_color = np.concatenate(gt_pcd_color)
                gt_inst_pcd_color = np.concatenate(gt_inst_pcd_color)

        v = None
        if not visualization_minimal_mode:
            v = vis.Visualizer()

            v.add_points(
                "RGB Input",
                full_res_coords_vis,
                colors=original_colors_np,
                normals=original_normals_np,
                visible=True,
                point_size=point_size,
            )

            if backbone_features is not None:
                v.add_points(
                    "PCA",
                    full_res_coords_vis,
                    colors=backbone_features,
                    normals=original_normals_np,
                    visible=False,
                    point_size=point_size,
                )

            if "labels" in target_full and len(gt_pcd_pos_vis) > 0:
                v.add_points(
                    "Semantics (GT)",
                    gt_pcd_pos_vis,
                    colors=gt_pcd_color,
                    normals=gt_pcd_normals,
                    alpha=0.8,
                    visible=False,
                    point_size=point_size,
                )
                v.add_points(
                    "Instances (GT)",
                    gt_pcd_pos_vis,
                    colors=gt_inst_pcd_color,
                    normals=gt_pcd_normals,
                    alpha=0.8,
                    visible=False,
                    point_size=point_size,
                )

        if pred_vis is None:
            pred_vis = self._build_unique_prediction_visualization(
                full_res_coords_raw=full_res_coords_raw,
                full_res_coords_vis=full_res_coords_vis,
                original_normals_np=original_normals_np,
                sorted_masks=sorted_masks,
                sort_classes=sort_classes,
                sort_scores_values=sort_scores_values,
                sorted_heatmaps=sorted_heatmaps,
            )
        pred_semantic_ids = pred_vis["pred_semantic_ids"]
        pred_instance_ids = pred_vis["pred_instance_ids"]
        pred_coords_vis = pred_vis["coords_vis"]
        pred_coords_raw = pred_vis["coords_raw"]
        pred_normals = pred_vis["normals"]
        pred_sem_color = pred_vis["sem_colors"]
        pred_inst_color = pred_vis["inst_colors"]

        if (not visualization_minimal_mode) and pred_coords_vis.shape[0] > 0:
            v.add_points(
                "Semantics (Mask3D)",
                pred_coords_vis,
                colors=pred_sem_color,
                normals=pred_normals,
                visible=False,
                alpha=0.8,
                point_size=point_size,
            )
            v.add_points(
                "Instances (Mask3D)",
                pred_coords_vis,
                colors=pred_inst_color,
                normals=pred_normals,
                visible=False,
                alpha=0.8,
                point_size=point_size,
            )

        if not visualization_minimal_mode:
            v.save(
                f"{self.config['general']['save_dir']}/visualizations/{file_name}"
            )

        ply_dir = (
            f"{self.config['general']['save_dir']}/visualizations/{file_name}"
        )
        os.makedirs(ply_dir, exist_ok=True)

        if pred_coords_vis.shape[0] > 0:
            write_rgb_ply(
                pred_coords_raw,
                pred_inst_color,
                f"{ply_dir}/pred_instances.ply",
            )
            write_rgb_ply(
                pred_coords_raw,
                pred_sem_color,
                f"{ply_dir}/pred_semantics.ply",
            )

        write_instance_semantic_ply(
            full_res_coords_raw,
            pred_semantic_ids,
            pred_instance_ids,
            f"{ply_dir}/pred_point_labels.ply",
            point_indices=point_indices,
            colors=original_colors_np,
        )

        if visualization_minimal_mode:
            return

        write_rgb_ply(
            full_res_coords_raw,
            original_colors_np,
            f"{ply_dir}/input_rgb.ply",
        )

        if "labels" in target_full and len(gt_pcd_pos_vis) > 0:
            write_rgb_ply(
                gt_pcd_pos_raw,
                gt_inst_pcd_color,
                f"{ply_dir}/gt_instances.ply",
            )
            write_rgb_ply(
                gt_pcd_pos_raw,
                gt_pcd_color,
                f"{ply_dir}/gt_semantics.ply",
            )
            write_instance_semantic_ply(
                full_res_coords_raw,
                gt_semantic_ids,
                gt_instance_ids,
                f"{ply_dir}/gt_point_labels.ply",
                point_indices=point_indices,
                colors=original_colors_np,
            )

    def eval_step(self, batch, batch_idx):
        data, target, file_names = batch
        inverse_maps = data.inverse_maps
        target_full = data.target_full
        original_colors = data.original_colors
        data_idx = data.idx
        original_normals = data.original_normals
        original_coordinates = data.original_coordinates

        # if len(target) == 0 or len(target_full) == 0:
        #    print("no targets")
        #    return None

        if len(data.coordinates) == 0:
            return 0.0

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]
        else:
            raw_coordinates = data.coordinates[:, 1:].float() * self.model.voxel_size

        if raw_coordinates.shape[0] == 0:
            return 0.0

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        try:
            output = self.forward(
                data,
                point2segment=[
                    target[i]["point2segment"] for i in range(len(target))
                ],
                raw_coordinates=raw_coordinates,
                is_eval=True,
            )
        except RuntimeError as run_err:
            print(run_err)
            if (
                "only a single point gives nans in cross-attention"
                == run_err.args[0]
            ):
                return None
            else:
                raise run_err

        if self.config.data.test_mode != "test":
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(False)

            try:
                losses = self.criterion(
                    output, target, mask_type=self.mask_type
                )
            except ValueError as val_err:
                print(f"ValueError: {val_err}")
                print(f"data shape: {data.shape}")
                print(f"data feat shape:  {data.features.shape}")
                print(f"data feat nans:   {data.features.isnan().sum()}")
                print(f"output: {output}")
                print(f"target: {target}")
                print(f"filenames: {file_names}")
                raise val_err

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(True)

        if self.config.general.save_visualizations:
            backbone_features = (
                output["backbone_features"].F.detach().cpu().numpy()
            )
            from sklearn import decomposition

            pca = decomposition.PCA(n_components=3)
            pca.fit(backbone_features)
            pca_features = pca.transform(backbone_features)
            rescaled_pca = (
                255
                * (pca_features - pca_features.min())
                / (pca_features.max() - pca_features.min())
            )

        self.eval_instance_step(
            output,
            target,
            target_full,
            inverse_maps,
            file_names,
            original_coordinates,
            original_colors,
            original_normals,
            raw_coordinates,
            data_idx,
            backbone_features=rescaled_pca
            if self.config.general.save_visualizations
            else None,
        )

        if self.config.data.test_mode != "test":
            return {
                f"val_{k}": v.detach().cpu().item() for k, v in losses.items()
            }
        else:
            return 0.0

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def get_full_res_mask(
        self, mask, inverse_map, point2segment_full, is_heatmap=False
    ):
        mask = mask.detach().cpu()[inverse_map]  # full res

        if self.eval_on_segments and is_heatmap == False:
            mask = scatter_mean(
                mask, point2segment_full, dim=0
            )  # full res segments
            mask = (mask > 0.5).float()
            mask = mask.detach().cpu()[
                point2segment_full.cpu()
            ]  # full res points

        return mask

    def get_mask_and_scores(
        self, mask_cls, mask_pred, iou_pred=None, num_queries=100, num_classes=18, device=None
    ):
        if device is None:
            device = self.device
        # Use the actual logits shape here. The eval path removes the no-object
        # channel before calling this helper, so relying on config/model class
        # counts can duplicate masks across pseudo classes.
        if mask_cls.ndim != 2:
            raise ValueError(f"mask_cls must be 2D, got shape={tuple(mask_cls.shape)}")
        num_queries = int(mask_cls.shape[0])
        num_classes = int(mask_cls.shape[1])
        if num_queries == 0 or num_classes == 0:
            empty_scores = torch.empty(0, device=device)
            empty_masks = torch.empty(
                (mask_pred.shape[0], 0), dtype=mask_pred.dtype, device=mask_pred.device
            )
            empty_classes = torch.empty(0, dtype=torch.long, device=device)
            empty_heatmap = torch.empty(
                (mask_pred.shape[0], 0), dtype=mask_pred.dtype, device=mask_pred.device
            )
            return empty_scores, empty_masks, empty_classes, empty_heatmap

        pre_topk_min_mask_points = int(
            getattr(self.config.general, "pre_topk_min_mask_points", 10)
        )
        if pre_topk_min_mask_points > 0:
            query_mask_sizes = (mask_pred > 0).sum(0)
            valid_queries = query_mask_sizes >= pre_topk_min_mask_points
            if not torch.any(valid_queries):
                empty_scores = torch.empty(0, device=device)
                empty_masks = torch.empty(
                    (mask_pred.shape[0], 0),
                    dtype=mask_pred.dtype,
                    device=mask_pred.device,
                )
                empty_classes = torch.empty(
                    0, dtype=torch.long, device=device
                )
                empty_heatmap = torch.empty(
                    (mask_pred.shape[0], 0),
                    dtype=mask_pred.dtype,
                    device=mask_pred.device,
                )
                return empty_scores, empty_masks, empty_classes, empty_heatmap
            mask_cls = mask_cls.clone()
            mask_cls[~valid_queries, :] = -float("inf")

        labels = (
            torch.arange(num_classes, device=device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        total_candidates = int(mask_cls.numel())
        if self.config.general.topk_per_image != -1:
            topk_k = min(int(self.config.general.topk_per_image), total_candidates)
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                topk_k, sorted=True
            )
        else:
            topk_k = min(num_queries, total_candidates)
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                topk_k, sorted=True
            )

        finite_keep = torch.isfinite(scores_per_query)
        scores_per_query = scores_per_query[finite_keep]
        topk_indices = topk_indices[finite_keep]
        if scores_per_query.numel() == 0:
            empty_scores = torch.empty(0, device=device)
            empty_masks = torch.empty(
                (mask_pred.shape[0], 0),
                dtype=mask_pred.dtype,
                device=mask_pred.device,
            )
            empty_classes = torch.empty(0, dtype=torch.long, device=device)
            empty_heatmap = torch.empty(
                (mask_pred.shape[0], 0),
                dtype=mask_pred.dtype,
                device=mask_pred.device,
            )
            return empty_scores, empty_masks, empty_classes, empty_heatmap

        labels_per_query = labels[topk_indices]
        topk_indices = torch.div(topk_indices, num_classes, rounding_mode="trunc")
        mask_pred = mask_pred[:, topk_indices]

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
            result_pred_mask.sum(0) + 1e-6
        )
        use_iou = getattr(self.config.general, "iou_head_loss_weight", 0.0) > 0
        if use_iou and iou_pred is not None:
            iou_factor = iou_pred.flatten()[topk_indices]
        else:
            iou_factor = torch.ones_like(mask_scores_per_image)
        score = scores_per_query * mask_scores_per_image * iou_factor
        classes = labels_per_query

        return score, result_pred_mask, classes, heatmap

    def eval_instance_step(
        self,
        output,
        target_low_res,
        target_full_res,
        inverse_maps,
        file_names,
        full_res_coords,
        original_colors,
        original_normals,
        raw_coords,
        idx,
        first_full_res=False,
        backbone_features=None,
    ):
        label_offset = self.validation_dataset.label_offset
        prediction = output["aux_outputs"]
        prediction.append(
            {
                "pred_logits": output["pred_logits"],
                "pred_masks": output["pred_masks"],
                "pred_iou": output.get("pred_iou", None),
            }
        )

        prediction[self.decoder_id][
            "pred_logits"
        ] = torch.functional.F.softmax(
            prediction[self.decoder_id]["pred_logits"], dim=-1
        )[
            ..., :-1
        ]

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()
        all_query_pos = list()

        offset_coords_idx = 0
        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            if not first_full_res:
                if self.model.train_on_segments:
                    masks = (
                        prediction[self.decoder_id]["pred_masks"][bid]
                        .detach()
                        .cpu()[target_low_res[bid]["point2segment"].cpu()]
                    )
                else:
                    masks = (
                        prediction[self.decoder_id]["pred_masks"][bid]
                        .detach()
                        .cpu()
                    )

                if self.config.general.use_dbscan:
                    new_preds = {
                        "pred_masks": list(),
                        "pred_logits": list(),
                        "pred_iou": list(),
                    }

                    curr_coords_idx = masks.shape[0]
                    curr_coords = raw_coords[
                        offset_coords_idx : curr_coords_idx + offset_coords_idx
                    ]
                    offset_coords_idx += curr_coords_idx

                    for curr_query in range(masks.shape[1]):
                        curr_masks = masks[:, curr_query] > 0

                        if curr_coords[curr_masks].shape[0] > 0:
                            clusters = (
                                DBSCAN(
                                    eps=self.config.general.dbscan_eps,
                                    min_samples=self.config.general.dbscan_min_points,
                                    n_jobs=-1,
                                )
                                .fit(curr_coords[curr_masks])
                                .labels_
                            )

                            new_mask = torch.zeros(curr_masks.shape, dtype=int)
                            new_mask[curr_masks] = (
                                torch.from_numpy(clusters) + 1
                            )

                            for cluster_id in np.unique(clusters):
                                original_pred_masks = masks[:, curr_query]
                                if cluster_id != -1:
                                    new_preds["pred_masks"].append(
                                        original_pred_masks
                                        * (new_mask == cluster_id + 1)
                                    )
                                    new_preds["pred_logits"].append(
                                        prediction[self.decoder_id][
                                            "pred_logits"
                                        ][bid, curr_query]
                                    )
                                    if (
                                        "pred_iou" in prediction[self.decoder_id]
                                        and prediction[self.decoder_id]["pred_iou"] is not None
                                    ):
                                        new_preds["pred_iou"].append(
                                            prediction[self.decoder_id]["pred_iou"][bid, curr_query]
                                        )

                    iou_pred = (
                        torch.stack(new_preds["pred_iou"]).cpu()
                        if ("pred_iou" in new_preds) and (len(new_preds["pred_iou"]) > 0)
                        else None
                    )
                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                        torch.stack(new_preds["pred_logits"]).cpu(),
                        torch.stack(new_preds["pred_masks"]).T,
                        iou_pred,
                        len(new_preds["pred_logits"]),
                        self.model.num_classes,
                    )
                else:
                    iou_pred = (
                        prediction[self.decoder_id]["pred_iou"][bid].detach().cpu()
                        if ("pred_iou" in prediction[self.decoder_id]) and (prediction[self.decoder_id]["pred_iou"] is not None)
                        else None
                    )
                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                        prediction[self.decoder_id]["pred_logits"][bid]
                        .detach()
                        .cpu(),
                        masks,
                        iou_pred,
                        prediction[self.decoder_id]["pred_logits"][bid].shape[
                            0
                        ],
                        self.model.num_classes,
                    )

                masks = self.get_full_res_mask(
                    masks,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                heatmap = self.get_full_res_mask(
                    heatmap,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                    is_heatmap=True,
                )

                if backbone_features is not None:
                    backbone_features = self.get_full_res_mask(
                        torch.from_numpy(backbone_features),
                        inverse_maps[bid],
                        target_full_res[bid]["point2segment"],
                        is_heatmap=True,
                    )
                    backbone_features = backbone_features.numpy()
            else:
                assert False, "not tested"
                masks = self.get_full_res_mask(
                    prediction[self.decoder_id]["pred_masks"][bid].cpu(),
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                scores, masks, classes, heatmap = self.get_mask_and_scores(
                    prediction[self.decoder_id]["pred_logits"][bid].cpu(),
                    masks,
                    None,
                    prediction[self.decoder_id]["pred_logits"][bid].shape[0],
                    self.model.num_classes,
                    device="cpu",
                )

            masks = masks.numpy()
            heatmap = heatmap.numpy()

            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            sort_scores_values = sort_scores.values.cpu().numpy()
            sort_classes = classes[sort_scores_index]

            sorted_masks = masks[:, sort_scores_index]
            sorted_heatmap = heatmap[:, sort_scores_index]

            if self.config.general.filter_out_instances:
                keep_instances = set()
                pairwise_overlap = sorted_masks.T @ sorted_masks
                normalization = pairwise_overlap.max(axis=0)
                norm_overlaps = pairwise_overlap / normalization

                for instance_id in range(norm_overlaps.shape[0]):
                    # filter out unlikely masks and nearly empty masks
                    # if not(sort_scores_values[instance_id] < 0.3 or sorted_masks[:, instance_id].sum() < 500):
                    if not (
                        sort_scores_values[instance_id]
                        < self.config.general.scores_threshold
                    ):
                        # check if mask != empty
                        if not sorted_masks[:, instance_id].sum() == 0.0:
                            overlap_ids = set(
                                np.nonzero(
                                    norm_overlaps[instance_id, :]
                                    > self.config.general.iou_threshold
                                )[0]
                            )

                            if len(overlap_ids) == 0:
                                keep_instances.add(instance_id)
                            else:
                                if instance_id == min(overlap_ids):
                                    keep_instances.add(instance_id)

                keep_instances = sorted(list(keep_instances))
                all_pred_classes.append(sort_classes[keep_instances])
                all_pred_masks.append(sorted_masks[:, keep_instances])
                all_pred_scores.append(sort_scores_values[keep_instances])
                all_heatmaps.append(sorted_heatmap[:, keep_instances])
            else:
                all_pred_classes.append(sort_classes)
                all_pred_masks.append(sorted_masks)
                all_pred_scores.append(sort_scores_values)
                all_heatmaps.append(sorted_heatmap)

        if self.validation_dataset.dataset_name == "scannet200":
            all_pred_classes[bid][all_pred_classes[bid] == 0] = -1
            if self.config.data.test_mode != "test":
                target_full_res[bid]["labels"][
                    target_full_res[bid]["labels"] == 0
                ] = -1

        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            pc_bid = all_pred_classes[bid]
            if not isinstance(pc_bid, np.ndarray):
                pc_bid = pc_bid.cpu()
            
            all_pred_classes[bid] = self.validation_dataset._remap_model_output(
                pc_bid + label_offset
            )

            if (
                self.config.data.test_mode != "test"
                and len(target_full_res) != 0
            ):
                labels_bid = target_full_res[bid]["labels"]
                if not isinstance(labels_bid, np.ndarray):
                    labels_bid = labels_bid.cpu()
                
                target_full_res[bid][
                    "labels"
                ] = self.validation_dataset._remap_model_output(
                    labels_bid + label_offset
                )

                # PREDICTION BOX
                bbox_data = []
                for query_id in range(
                    all_pred_masks[bid].shape[1]
                ):  # self.model.num_queries
                    obj_coords = full_res_coords[bid][
                        all_pred_masks[bid][:, query_id].astype(bool), :
                    ]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))

                        bbox_data.append(
                            (
                                all_pred_classes[bid][query_id].item(),
                                bbox,
                                all_pred_scores[bid][query_id],
                            )
                        )
                self.bbox_preds[file_names[bid]] = bbox_data

                # GT BOX
                bbox_data = []
                for obj_id in range(target_full_res[bid]["masks"].shape[0]):
                    if target_full_res[bid]["labels"][obj_id].item() == 255:
                        continue

                    curr_mask = target_full_res[bid]["masks"][obj_id, :]
                    if not isinstance(curr_mask, np.ndarray):
                        curr_mask = curr_mask.cpu().detach().numpy()
                    
                    obj_coords = full_res_coords[bid][
                        curr_mask.astype(bool),
                        :,
                    ]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))
                        bbox_data.append(
                            (
                                target_full_res[bid]["labels"][obj_id].item(),
                                bbox,
                            )
                        )

                self.bbox_gt[file_names[bid]] = bbox_data

            if self.config.general.eval_inner_core == -1:
                coords_eval = np.asarray(full_res_coords[bid], dtype=np.float32)
                normals_eval = np.asarray(original_normals[bid])
                masks_eval = all_pred_masks[bid]
                heatmaps_eval = all_heatmaps[bid]
            else:
                cond_inner = self.test_dataset.data[idx[bid]]["cond_inner"]
                coords_eval = np.asarray(
                    full_res_coords[bid][cond_inner], dtype=np.float32
                )
                normals_eval = np.asarray(original_normals[bid][cond_inner])
                masks_eval = all_pred_masks[bid][cond_inner]
                heatmaps_eval = all_heatmaps[bid][cond_inner]

            coords_vis_eval = coords_eval.copy()
            coords_vis_eval -= coords_vis_eval.mean(axis=0)
            postprocessed_pred = self._build_unique_prediction_visualization(
                full_res_coords_raw=coords_eval,
                full_res_coords_vis=coords_vis_eval,
                original_normals_np=normals_eval,
                sorted_masks=[masks_eval],
                sort_classes=[all_pred_classes[bid]],
                sort_scores_values=[all_pred_scores[bid]],
                sorted_heatmaps=[heatmaps_eval],
            )

            self.preds[file_names[bid]] = {
                "pred_masks": np.asarray(
                    postprocessed_pred["pred_masks"], dtype=np.uint8
                ),
                "pred_scores": np.asarray(
                    postprocessed_pred["pred_scores"], dtype=np.float32
                ),
                "pred_classes": np.asarray(
                    postprocessed_pred["pred_classes"], dtype=np.int32
                ),
            }

            if (
                self.config.data.test_mode != "test"
                and len(target_full_res) != 0
            ):
                bbox_data = []
                for query_id in range(self.preds[file_names[bid]]["pred_masks"].shape[1]):
                    obj_coords = coords_eval[
                        self.preds[file_names[bid]]["pred_masks"][
                            :, query_id
                        ].astype(bool),
                        :,
                    ]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(axis=0) - obj_coords.min(
                            axis=0
                        )
                        bbox = np.concatenate((obj_center, obj_axis_length))
                        bbox_data.append(
                            (
                                int(
                                    self.preds[file_names[bid]]["pred_classes"][
                                        query_id
                                    ]
                                ),
                                bbox,
                                float(
                                    self.preds[file_names[bid]]["pred_scores"][
                                        query_id
                                    ]
                                ),
                            )
                        )
                self.bbox_preds[file_names[bid]] = bbox_data

            # accumulate semantic confusion for mIoU
            N_points = full_res_coords[bid].shape[0]
            pred_sem = np.zeros(N_points, dtype=np.int32)
            pc = self.preds[file_names[bid]]["pred_classes"]
            if not isinstance(pc, np.ndarray):
                pc = pc.cpu().numpy()
            
            ps = self.preds[file_names[bid]]["pred_scores"]
            if not isinstance(ps, np.ndarray):
                ps = ps.cpu().numpy()
            
            pm = self.preds[file_names[bid]]["pred_masks"]
            if not isinstance(pm, np.ndarray):
                pm = pm.cpu().numpy()
            if pm.shape[1] > 0:
                order = np.argsort(ps)[::-1]
                occ = np.zeros(N_points, dtype=bool)
                for k in order:
                    mk = pm[:, k].astype(bool)
                    sel = mk & (~occ)
                    if np.any(sel):
                        curr_class_id = int(pc[k])
                        # 核心：如果 ID 合法（包括 0），则记录
                        if 0 <= curr_class_id < self.model.num_classes:
                            pred_sem[sel] = curr_class_id
                        occ[sel] = True
            gt_sem = np.zeros(N_points, dtype=np.int32)
            if "labels" in target_full_res[bid]:
                for obj_id in range(target_full_res[bid]["masks"].shape[0]):
                    lbl = int(target_full_res[bid]["labels"][obj_id].item())
                    if lbl == 255:
                        continue
                    
                    curr_mask = target_full_res[bid]["masks"][obj_id, :]
                    if not isinstance(curr_mask, np.ndarray):
                        curr_mask = curr_mask.cpu().numpy()
                    
                    mask_np = curr_mask.astype(bool)
                    gt_sem[mask_np] = lbl

            pred_sem = np.zeros(coords_eval.shape[0], dtype=np.int32)
            valid_pred_sem = postprocessed_pred["pred_semantic_ids"] >= 0
            pred_sem[valid_pred_sem] = postprocessed_pred["pred_semantic_ids"][
                valid_pred_sem
            ]

            gt_sem = np.zeros(coords_eval.shape[0], dtype=np.int32)
            target_conf = target_full_res[bid]
            if self.config.general.eval_inner_core != -1:
                target_conf = dict(target_full_res[bid])
                target_conf["masks"] = target_full_res[bid]["masks"][:, cond_inner]
            if "labels" in target_conf:
                for obj_id in range(target_conf["masks"].shape[0]):
                    lbl = int(target_conf["labels"][obj_id].item())
                    if lbl == 255:
                        continue
                    curr_mask = target_conf["masks"][obj_id, :]
                    if not isinstance(curr_mask, np.ndarray):
                        curr_mask = curr_mask.cpu().numpy()
                    gt_sem[curr_mask.astype(bool)] = lbl
            self.confusion.add(pred_sem, gt_sem)

            if self.config.general.save_visualizations:
                if "cond_inner" in self.test_dataset.data[idx[bid]]:
                    cond_inner = self.test_dataset.data[idx[bid]]["cond_inner"]
                    cond_inner_np = to_numpy_array(cond_inner)
                    if cond_inner_np.dtype == bool:
                        point_indices = np.where(cond_inner_np)[0].astype(np.int32)
                    else:
                        point_indices = cond_inner_np.astype(np.int32)
                    target_full_res[bid]["masks"] = target_full_res[bid][
                        "masks"
                    ][:, cond_inner]
                    self.save_visualizations(
                        target_full_res[bid],
                        full_res_coords[bid][cond_inner],
                        [self.preds[file_names[bid]]["pred_masks"]],
                        [self.preds[file_names[bid]]["pred_classes"]],
                        file_names[bid],
                        original_colors[bid][cond_inner],
                        original_normals[bid][cond_inner],
                        [self.preds[file_names[bid]]["pred_scores"]],
                        sorted_heatmaps=[
                            all_heatmaps[bid][cond_inner]
                        ],
                        query_pos=all_query_pos[bid][cond_inner]
                        if len(all_query_pos) > 0
                        else None,
                        backbone_features=backbone_features[cond_inner],
                        point_size=self.config.general.visualization_point_size,
                        point_indices=point_indices,
                        pred_vis=postprocessed_pred,
                    )
                else:
                    self.save_visualizations(
                        target_full_res[bid],
                        full_res_coords[bid],
                        [self.preds[file_names[bid]]["pred_masks"]],
                        [self.preds[file_names[bid]]["pred_classes"]],
                        file_names[bid],
                        original_colors[bid],
                        original_normals[bid],
                        [self.preds[file_names[bid]]["pred_scores"]],
                        sorted_heatmaps=[all_heatmaps[bid]],
                        query_pos=all_query_pos[bid]
                        if len(all_query_pos) > 0
                        else None,
                        backbone_features=backbone_features,
                        point_size=self.config.general.visualization_point_size,
                        point_indices=np.arange(
                            full_res_coords[bid].shape[0], dtype=np.int32
                        ),
                        pred_vis=postprocessed_pred,
                    )

            if self.config.general.export:
                if self.validation_dataset.dataset_name == "stpls3d":
                    scan_id, _, _, crop_id = file_names[bid].split("_")
                    crop_id = int(crop_id.replace(".txt", ""))
                    file_name = (
                        f"{scan_id}_points_GTv3_0{crop_id}_inst_nostuff"
                    )

                    self.export(
                        self.preds[file_names[bid]]["pred_masks"],
                        self.preds[file_names[bid]]["pred_scores"],
                        self.preds[file_names[bid]]["pred_classes"],
                        file_name,
                        self.decoder_id,
                    )
                else:
                    self.export(
                        self.preds[file_names[bid]]["pred_masks"],
                        self.preds[file_names[bid]]["pred_scores"],
                        self.preds[file_names[bid]]["pred_classes"],
                        file_names[bid],
                        self.decoder_id,
                    )

    def eval_instance_epoch_end(self):
        log_prefix = f"val"
        ap_results = {}

        head_results, tail_results, common_results = [], [], []

        total_gt_boxes = sum(len(v) for v in self.bbox_gt.values())
        total_pred_boxes = sum(len(v) for v in self.bbox_preds.values())
        gt_box_class_counts = defaultdict(int)
        pred_box_class_counts = defaultdict(int)
        for _, items in self.bbox_gt.items():
            for cls_id, _bbox in items:
                gt_box_class_counts[int(cls_id)] += 1
        for _, items in self.bbox_preds.items():
            for cls_id, _bbox, _score in items:
                pred_box_class_counts[int(cls_id)] += 1

        def _name_map(counter):
            out = {}
            for cls_id, cnt in sorted(counter.items()):
                name = self.train_dataset.label_info.get(cls_id, {"name": str(cls_id)})["name"]
                out[name] = cnt
            return out

        print(
            f"[Eval] total_bbox_gt={total_gt_boxes} total_bbox_pred={total_pred_boxes}"
        )
        print(f"[Eval] gt_bbox_by_class={_name_map(gt_box_class_counts)}")
        print(f"[Eval] pred_bbox_by_class={_name_map(pred_box_class_counts)}")

        box_ap_50 = eval_det(
            self.bbox_preds, self.bbox_gt, ovthresh=0.5, use_07_metric=False
        )
        box_ap_25 = eval_det(
            self.bbox_preds, self.bbox_gt, ovthresh=0.25, use_07_metric=False
        )
        mean_box_ap_25 = sum([v for k, v in box_ap_25[-1].items()]) / len(
            box_ap_25[-1].keys()
        )
        mean_box_ap_50 = sum([v for k, v in box_ap_50[-1].items()]) / len(
            box_ap_50[-1].keys()
        )

        ap_results[f"{log_prefix}_mean_box_ap_25"] = mean_box_ap_25
        ap_results[f"{log_prefix}_mean_box_ap_50"] = mean_box_ap_50

        for class_id in box_ap_50[-1].keys():
            if class_id in self.train_dataset.label_info:
                class_name = self.train_dataset.label_info[class_id]["name"]
                ap_results[f"{log_prefix}_{class_name}_val_box_ap_50"] = box_ap_50[
                    -1
                ][class_id]

        for class_id in box_ap_25[-1].keys():
            if class_id in self.train_dataset.label_info:
                class_name = self.train_dataset.label_info[class_id]["name"]
                ap_results[f"{log_prefix}_{class_name}_val_box_ap_25"] = box_ap_25[
                    -1
                ][class_id]

        root_path = f"eval_output"
        base_path = f"{root_path}/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}"

        if self.validation_dataset.dataset_name in [
            "scannet",
            "stpls3d",
            "scannet200",
            "pheno4d",
            "pheno4d_maize",
            "arabidopsis",
        ]:
            gt_data_path = f"{self.validation_dataset.data_dir[0]}/instance_gt/{self.validation_dataset.mode}"
        else:
            gt_data_path = f"{self.validation_dataset.data_dir[0]}/instance_gt/Area_{self.config.general.area}"

        pred_path = f"{base_path}/tmp_output.txt"

        log_prefix = f"val"

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        try:
            print(
                f"[Eval] epoch={self.current_epoch} dataset={self.validation_dataset.dataset_name} "
                f"pred_scans={len(self.preds)} gt_path={gt_data_path}"
            )
            if os.path.isdir(gt_data_path):
                gt_file_count = len(list(Path(gt_data_path).glob("*.txt")))
                print(f"[Eval] gt_files={gt_file_count}")
            else:
                print(f"[Eval][WARN] GT path does not exist: {gt_data_path}")

            pred_keys = list(self.preds.keys())
            if len(pred_keys) > 0:
                print(
                    f"[Eval] sample_pred_keys(first 5/{len(pred_keys)})={pred_keys[:5]}"
                )
                if len(pred_keys) <= 50:
                    print(f"[Eval] all_pred_keys={pred_keys}")

            bad_shape_summaries = []
            for scan_key, scan_pred in self.preds.items():
                pred_masks = scan_pred.get("pred_masks", None)
                pred_scores = scan_pred.get("pred_scores", None)
                pred_classes = scan_pred.get("pred_classes", None)

                masks_shape = getattr(pred_masks, "shape", None)
                n_masks = -1
                if masks_shape is not None and len(masks_shape) >= 2:
                    n_masks = int(masks_shape[1])
                n_scores = len(pred_scores) if pred_scores is not None else -1
                n_classes = len(pred_classes) if pred_classes is not None else -1

                if n_masks != n_scores or n_masks != n_classes:
                    bad_shape_summaries.append(
                        f"scan={scan_key} masks_shape={masks_shape} "
                        f"n_masks={n_masks} n_scores={n_scores} n_classes={n_classes}"
                    )
            if bad_shape_summaries:
                print("[Eval][ERROR] Inconsistent prediction tensor shapes detected:")
                for item in bad_shape_summaries[:10]:
                    print(f"[Eval][ERROR] {item}")
                raise IndexError(
                    "Prediction shapes are inconsistent; see [Eval][ERROR] details above."
                )

            if self.validation_dataset.dataset_name == "s3dis":
                new_preds = {}
                for key in self.preds.keys():
                    new_preds[
                        key.replace(f"Area_{self.config.general.area}_", "")
                    ] = {
                        "pred_classes": self.preds[key]["pred_classes"] + 1,
                        "pred_masks": self.preds[key]["pred_masks"],
                        "pred_scores": self.preds[key]["pred_scores"],
                    }
                mprec, mrec = evaluate(
                    new_preds, gt_data_path, pred_path, dataset="s3dis"
                )
                ap_results[f"{log_prefix}_mean_precision"] = mprec
                ap_results[f"{log_prefix}_mean_recall"] = mrec
            elif self.validation_dataset.dataset_name == "stpls3d":
                new_preds = {}
                for key in self.preds.keys():
                    new_preds[key.replace(".txt", "")] = {
                        "pred_classes": self.preds[key]["pred_classes"],
                        "pred_masks": self.preds[key]["pred_masks"],
                        "pred_scores": self.preds[key]["pred_scores"],
                    }

                evaluate(new_preds, gt_data_path, pred_path, dataset="stpls3d")
            elif self.validation_dataset.dataset_name == "arabidopsis":
                evaluate(self.preds, gt_data_path, pred_path, dataset="arabidopsis")
            else:
                evaluate(
                    self.preds, gt_data_path, pred_path, dataset="scannet"
                )

            if not os.path.isfile(pred_path):
                raise OSError(f"Evaluation output file was not created: {pred_path}")
            pred_size = os.path.getsize(pred_path)
            print(f"[Eval] result_file={pred_path} size={pred_size} bytes")

            parsed_rows = 0
            with open(pred_path, "r") as fin:
                for line_id, line in enumerate(fin):
                    if line_id == 0:
                        # ignore header
                        continue
                    class_name, _, ap, ap_50, ap_25 = line.strip().split(",")
                    parsed_rows += 1

                    if self.validation_dataset.dataset_name == "scannet200":
                        if class_name in VALID_CLASS_IDS_200_VALIDATION:
                            ap_results[
                                f"{log_prefix}_{class_name}_val_ap"
                            ] = float(ap)
                            ap_results[
                                f"{log_prefix}_{class_name}_val_ap_50"
                            ] = float(ap_50)
                            ap_results[
                                f"{log_prefix}_{class_name}_val_ap_25"
                            ] = float(ap_25)

                            if class_name in HEAD_CATS_SCANNET_200:
                                head_results.append(
                                    np.array(
                                        (float(ap), float(ap_50), float(ap_25))
                                    )
                                )
                            elif class_name in COMMON_CATS_SCANNET_200:
                                common_results.append(
                                    np.array(
                                        (float(ap), float(ap_50), float(ap_25))
                                    )
                                )
                            elif class_name in TAIL_CATS_SCANNET_200:
                                tail_results.append(
                                    np.array(
                                        (float(ap), float(ap_50), float(ap_25))
                                    )
                                )
                            else:
                                assert (False, "class not known!")
                    else:
                        ap_results[
                            f"{log_prefix}_{class_name}_val_ap"
                        ] = float(ap)
                        ap_results[
                            f"{log_prefix}_{class_name}_val_ap_50"
                        ] = float(ap_50)
                        ap_results[
                            f"{log_prefix}_{class_name}_val_ap_25"
                        ] = float(ap_25)
            print(f"[Eval] parsed_rows={parsed_rows}")

            if self.validation_dataset.dataset_name == "scannet200":
                head_results = np.stack(head_results)
                common_results = np.stack(common_results)
                tail_results = np.stack(tail_results)

                mean_tail_results = np.nanmean(tail_results, axis=0)
                mean_common_results = np.nanmean(common_results, axis=0)
                mean_head_results = np.nanmean(head_results, axis=0)

                ap_results[
                    f"{log_prefix}_mean_tail_ap_25"
                ] = mean_tail_results[0]
                ap_results[
                    f"{log_prefix}_mean_common_ap_25"
                ] = mean_common_results[0]
                ap_results[
                    f"{log_prefix}_mean_head_ap_25"
                ] = mean_head_results[0]

                ap_results[
                    f"{log_prefix}_mean_tail_ap_50"
                ] = mean_tail_results[1]
                ap_results[
                    f"{log_prefix}_mean_common_ap_50"
                ] = mean_common_results[1]
                ap_results[
                    f"{log_prefix}_mean_head_ap_50"
                ] = mean_head_results[1]

                ap_results[
                    f"{log_prefix}_mean_tail_ap_25"
                ] = mean_tail_results[2]
                ap_results[
                    f"{log_prefix}_mean_common_ap_25"
                ] = mean_common_results[2]
                ap_results[
                    f"{log_prefix}_mean_head_ap_25"
                ] = mean_head_results[2]

                overall_ap_results = np.nanmean(
                    np.vstack((head_results, common_results, tail_results)),
                    axis=0,
                )

                ap_results[f"{log_prefix}_mean_ap"] = overall_ap_results[0]
                ap_results[f"{log_prefix}_mean_ap_50"] = overall_ap_results[1]
                ap_results[f"{log_prefix}_mean_ap_25"] = overall_ap_results[2]

                ap_results = {
                    key: 0.0 if math.isnan(score) else score
                    for key, score in ap_results.items()
                }
            else:
                mean_ap = statistics.mean(
                    [
                        item
                        for key, item in ap_results.items()
                        if key.endswith("val_ap")
                    ]
                )
                mean_ap_50 = statistics.mean(
                    [
                        item
                        for key, item in ap_results.items()
                        if key.endswith("val_ap_50")
                    ]
                )
                mean_ap_25 = statistics.mean(
                    [
                        item
                        for key, item in ap_results.items()
                        if key.endswith("val_ap_25")
                    ]
                )

                ap_results[f"{log_prefix}_mean_ap"] = mean_ap
                ap_results[f"{log_prefix}_mean_ap_50"] = mean_ap_50
                ap_results[f"{log_prefix}_mean_ap_25"] = mean_ap_25

                ap_results = {
                    key: 0.0 if math.isnan(score) else score
                    for key, score in ap_results.items()
                }
        except (IndexError, OSError) as e:
            print("NO SCORES!!!")
            print(
                f"[Eval][ERROR] epoch={self.current_epoch} dataset={self.validation_dataset.dataset_name} "
                f"pred_path={pred_path}"
            )
            print(f"[Eval][ERROR] {type(e).__name__}: {e}")
            print(traceback.format_exc())
            ap_results[f"{log_prefix}_mean_ap"] = 0.0
            ap_results[f"{log_prefix}_mean_ap_50"] = 0.0
            ap_results[f"{log_prefix}_mean_ap_25"] = 0.0

        # semantic mIoU
        conf_mat = self.confusion.value()
        iou_per_class = self.iou.value(conf_mat)
        miou = float(np.nanmean(iou_per_class)) if len(iou_per_class) > 0 else 0.0
        ap_results[f"{log_prefix}_mean_miou"] = miou
        for cid in range(len(iou_per_class)):
            cname = self.train_dataset.label_info.get(cid, {"name": str(cid)})["name"]
            ap_results[f"{log_prefix}_{cname}_miou"] = float(iou_per_class[cid])
        self.confusion.reset()

        print("==== Validation Metrics ====")
        print(f"mean_ap: {ap_results.get(f'{log_prefix}_mean_ap', float('nan')):.3f} | ap50: {ap_results.get(f'{log_prefix}_mean_ap_50', float('nan')):.3f} | ap25: {ap_results.get(f'{log_prefix}_mean_ap_25', float('nan')):.3f}")
        if f"{log_prefix}_mean_miou" in ap_results:
            print(f"mean_miou: {ap_results[f'{log_prefix}_mean_miou']:.3f}")
        cls_ap = sorted([k for k in ap_results.keys() if k.endswith('val_ap')])
        for k in cls_ap:
            print(f"{k}: {ap_results[k]:.3f}")
        cls_miou = sorted([k for k in ap_results.keys() if k.endswith('_miou') and not k.endswith('mean_miou')])
        for k in cls_miou:
            print(f"{k}: {ap_results[k]:.3f}")

        self.log_dict(ap_results)

        if not self.config.general.export:
            shutil.rmtree(base_path)

        del self.preds
        del self.bbox_preds
        del self.bbox_gt

        gc.collect()

        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()

    def test_epoch_end(self, outputs):
        if self.config.general.export:
            return

        self.eval_instance_epoch_end()

        dd = defaultdict(list)
        for output in outputs:
            for key, val in output.items():  # .items() in Python 3.
                dd[key].append(val)

        dd = {k: statistics.mean(v) for k, v in dd.items()}

        dd["val_mean_loss_ce"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_ce" in k]]
        )
        dd["val_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_mask" in k]]
        )
        dd["val_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_dice" in k]]
        )

        self.log_dict(dd)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(
            self.config.data.train_dataset
        )
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(
            self.config.data.test_dataset
        )
        self.labels_info = self.train_dataset.label_info

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )
