from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    import pandas as pd
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.spatial import Delaunay, QhullError, cKDTree
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependencies. Run this script in a Python environment with "
        "numpy, scipy, scikit-learn, and pandas installed."
    ) from exc


REQUIRED_COLUMNS = [
    "x",
    "y",
    "z",
    "semantic_id",
    "instance_id",
    "point_index",
]
MIN_POINTS_PER_LEAF = 100
KNN_OUTLIER_NEIGHBORS = 8
GRAPH_NEIGHBORS = 12
WIDTH_SECTION_COUNT = 15
PREDECESSOR_SENTINEL = -9999
MIN_STEM_SEED_POINTS = 15
NO_CONTACT_SEED_FRACTION = 0.03
NO_CONTACT_SEED_MAX_POINTS = 200
NO_CONTACT_DISTANCE_MARGIN = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Arabidopsis leaf traits from pred_point_labels.ply. "
            "Only pred_point_labels.ply is used for numeric computation."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to pred_point_labels.ply.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for leaf_traits.csv, plant_summary.json, and leaf_debug.csv.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Coordinate scaling factor applied before trait computation.",
    )
    return parser.parse_args()


def parse_ascii_ply_header(filepath: Path) -> Tuple[int, List[str], int]:
    property_names: List[str] = []
    vertex_count: Optional[int] = None
    header_lines = 0
    inside_vertex_element = False

    with filepath.open("r", encoding="ascii", errors="ignore") as handle:
        first_line = handle.readline().strip()
        header_lines += 1
        if first_line != "ply":
            raise ValueError(f"{filepath} is not a PLY file.")

        format_line = handle.readline().strip()
        header_lines += 1
        if format_line != "format ascii 1.0":
            raise ValueError(
                f"{filepath} must be an ASCII PLY exported by this project."
            )

        for line in handle:
            header_lines += 1
            stripped = line.strip()

            if stripped.startswith("element "):
                parts = stripped.split()
                inside_vertex_element = len(parts) >= 3 and parts[1] == "vertex"
                if inside_vertex_element:
                    vertex_count = int(parts[2])
                continue

            if stripped.startswith("property ") and inside_vertex_element:
                property_names.append(stripped.split()[-1])
                continue

            if stripped == "end_header":
                break

    if vertex_count is None:
        raise ValueError(f"{filepath} does not contain a vertex element.")

    return header_lines, property_names, vertex_count


def load_prediction_ply(filepath: Path, scale: float) -> pd.DataFrame:
    header_lines, property_names, vertex_count = parse_ascii_ply_header(filepath)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in property_names]
    if missing_columns:
        raise ValueError(
            f"{filepath} is missing required columns: {', '.join(missing_columns)}"
        )

    frame = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=header_lines,
        nrows=vertex_count,
        header=None,
        names=property_names,
        engine="python",
    )
    frame = frame[REQUIRED_COLUMNS].copy()
    frame[["x", "y", "z"]] = frame[["x", "y", "z"]] * float(scale)
    for column in ("semantic_id", "instance_id", "point_index"):
        frame[column] = frame[column].astype(np.int32)
    return frame


def estimate_pseudo_stem_points(
    leaf_frame: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, object]]:
    grouped_leaves = list(leaf_frame.groupby("instance_id", sort=True))
    if not grouped_leaves:
        raise ValueError("Cannot estimate a pseudo stem without valid leaf instances.")

    centroid_points: List[np.ndarray] = []
    low_z_values: List[float] = []
    per_leaf_points: List[np.ndarray] = []
    for _, group in grouped_leaves:
        points = group[["x", "y", "z"]].to_numpy(dtype=np.float64)
        centroid_points.append(points.mean(axis=0))
        low_z_values.append(float(np.percentile(points[:, 2], 10.0)))
        per_leaf_points.append(points)

    centroids = np.vstack(centroid_points)
    rough_center = np.array(
        [
            float(np.median(centroids[:, 0])),
            float(np.median(centroids[:, 1])),
            float(np.median(np.asarray(low_z_values, dtype=np.float64))),
        ],
        dtype=np.float64,
    )

    first_pass_candidates: List[np.ndarray] = []
    for points in per_leaf_points:
        distances = np.linalg.norm(points - rough_center[None, :], axis=1)
        first_pass_candidates.append(points[int(np.argmin(distances))])
    first_pass_candidates = np.vstack(first_pass_candidates)

    refined_center = np.median(first_pass_candidates, axis=0)
    second_pass_candidates: List[np.ndarray] = []
    for points in per_leaf_points:
        distances = np.linalg.norm(points - refined_center[None, :], axis=1)
        second_pass_candidates.append(points[int(np.argmin(distances))])
    second_pass_candidates = np.vstack(second_pass_candidates)

    pseudo_stem_points = np.vstack([second_pass_candidates, refined_center[None, :]])
    metadata = {
        "stem_source": "estimated_from_leaf_geometry",
        "estimated_stem_center": refined_center.tolist(),
        "estimated_stem_point_count": int(pseudo_stem_points.shape[0]),
        "estimated_stem_method": "median_leaf_centers_then_nearest_leaf_points",
    }
    return pseudo_stem_points, metadata


def median_absolute_deviation(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def polyline_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def polyline_cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if points.shape[0] == 1:
        return np.zeros((1,), dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(
        [np.zeros((1,), dtype=np.float64), np.cumsum(segment_lengths)]
    )


def normalize_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return vector / norm


def mean_neighbor_distance(points: np.ndarray, neighbors: int) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.zeros((points.shape[0],), dtype=np.float64)
    nn_count = min(neighbors + 1, points.shape[0])
    model = NearestNeighbors(n_neighbors=nn_count)
    model.fit(points)
    distances, _ = model.kneighbors(points)
    if distances.shape[1] <= 1:
        return np.zeros((points.shape[0],), dtype=np.float64)
    return distances[:, 1:].mean(axis=1)


def median_second_neighbor_distance(points: np.ndarray) -> float:
    if points.shape[0] <= 1:
        return 0.0

    nn_count = min(3, points.shape[0])
    model = NearestNeighbors(n_neighbors=nn_count)
    model.fit(points)
    distances, _ = model.kneighbors(points)
    if distances.shape[1] >= 3:
        values = distances[:, 2]
    else:
        values = distances[:, 1]

    positive_values = values[values > 0]
    if positive_values.size == 0:
        return 0.0
    return float(np.median(positive_values))


def build_knn_graph(
    points: np.ndarray,
    neighbors: int,
    max_edge_length: float,
) -> csr_matrix:
    if points.shape[0] <= 1:
        return csr_matrix((points.shape[0], points.shape[0]), dtype=np.float64)

    nn_count = min(neighbors + 1, points.shape[0])
    model = NearestNeighbors(n_neighbors=nn_count)
    model.fit(points)
    distances, indices = model.kneighbors(points)

    row_indices = np.repeat(np.arange(points.shape[0]), nn_count - 1)
    col_indices = indices[:, 1:].reshape(-1)
    weights = distances[:, 1:].reshape(-1)
    keep_mask = (weights > 0) & (weights <= max_edge_length)

    row_indices = row_indices[keep_mask]
    col_indices = col_indices[keep_mask]
    weights = weights[keep_mask]

    if weights.size == 0:
        return csr_matrix((points.shape[0], points.shape[0]), dtype=np.float64)

    data = np.concatenate([weights, weights])
    rows = np.concatenate([row_indices, col_indices])
    cols = np.concatenate([col_indices, row_indices])
    graph = coo_matrix((data, (rows, cols)), shape=(points.shape[0], points.shape[0]))
    graph = graph.tocsr()
    graph.sum_duplicates()
    return graph


def keep_largest_component(
    points: np.ndarray,
    point_indices: np.ndarray,
    graph: csr_matrix,
) -> Tuple[np.ndarray, np.ndarray, csr_matrix]:
    if points.shape[0] == 0:
        return points, point_indices, graph

    component_count, labels = connected_components(
        graph, directed=False, return_labels=True
    )
    if component_count <= 1:
        return points, point_indices, graph

    component_sizes = np.bincount(labels)
    largest_component = int(np.argmax(component_sizes))
    keep_mask = labels == largest_component
    return points[keep_mask], point_indices[keep_mask], graph[keep_mask][:, keep_mask]


def select_seed_cluster(
    candidate_indices: np.ndarray,
    graph: csr_matrix,
    distances_to_stem: np.ndarray,
) -> np.ndarray:
    if candidate_indices.size == 0:
        return candidate_indices

    candidate_indices = np.asarray(candidate_indices, dtype=np.int32)
    subgraph = graph[candidate_indices][:, candidate_indices]
    component_count, labels = connected_components(
        subgraph, directed=False, return_labels=True
    )
    if component_count <= 1:
        return candidate_indices

    best_indices = candidate_indices[:1]
    best_key = (-1, -math.inf)
    for component_id in range(component_count):
        component_mask = labels == component_id
        component_indices = candidate_indices[component_mask]
        size = int(component_indices.size)
        mean_neg_distance = -float(
            np.mean(distances_to_stem[component_indices])
        )
        key = (size, mean_neg_distance)
        if key > best_key:
            best_key = key
            best_indices = component_indices

    return np.asarray(best_indices, dtype=np.int32)


def multi_source_shortest_paths(graph: csr_matrix, seed_indices: np.ndarray) -> np.ndarray:
    if seed_indices.size == 0:
        return np.full((graph.shape[0],), np.inf, dtype=np.float64)
    distances = dijkstra(
        graph, directed=False, indices=seed_indices.astype(np.int32).tolist()
    )
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim == 1:
        return distances
    return distances.min(axis=0)


def reconstruct_path(
    predecessors: np.ndarray, source_index: int, target_index: int
) -> np.ndarray:
    path = [int(target_index)]
    current = int(target_index)
    while current != int(source_index):
        predecessor = int(predecessors[current])
        if predecessor == PREDECESSOR_SENTINEL:
            raise ValueError("No path exists between source and target.")
        path.append(predecessor)
        current = predecessor
    path.reverse()
    return np.asarray(path, dtype=np.int32)


def resample_polyline(points: np.ndarray, step: float) -> np.ndarray:
    if points.shape[0] <= 2:
        return points.copy()

    cumulative = polyline_cumulative_lengths(points)
    total_length = float(cumulative[-1])
    if total_length <= 0:
        return points.copy()

    sample_positions = np.arange(0.0, total_length, step, dtype=np.float64)
    if sample_positions.size == 0 or sample_positions[0] != 0.0:
        sample_positions = np.insert(sample_positions, 0, 0.0)
    if sample_positions[-1] < total_length:
        sample_positions = np.append(sample_positions, total_length)

    resampled = np.empty((sample_positions.shape[0], 3), dtype=np.float64)
    for axis in range(3):
        resampled[:, axis] = np.interp(
            sample_positions, cumulative, points[:, axis]
        )
    return resampled


def smooth_polyline(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 2:
        return points.copy()
    smoothed = points.copy()
    smoothed[1:-1] = (points[:-2] + points[1:-1] + points[2:]) / 3.0
    return smoothed


def sample_polyline_point(points: np.ndarray, arc_length: float) -> np.ndarray:
    cumulative = polyline_cumulative_lengths(points)
    total_length = float(cumulative[-1])
    if total_length <= 0 or points.shape[0] == 1:
        return points[0].copy()
    if arc_length <= 0:
        return points[0].copy()
    if arc_length >= total_length:
        return points[-1].copy()

    segment_index = int(np.searchsorted(cumulative, arc_length, side="right") - 1)
    segment_index = max(0, min(segment_index, points.shape[0] - 2))
    segment_length = float(cumulative[segment_index + 1] - cumulative[segment_index])
    if segment_length <= 0:
        return points[segment_index].copy()

    alpha = (arc_length - cumulative[segment_index]) / segment_length
    return (
        (1.0 - alpha) * points[segment_index]
        + alpha * points[segment_index + 1]
    )


def sample_polyline_tangent(points: np.ndarray, arc_length: float) -> Optional[np.ndarray]:
    if points.shape[0] < 2:
        return None

    cumulative = polyline_cumulative_lengths(points)
    total_length = float(cumulative[-1])
    if total_length <= 0:
        return normalize_vector(points[-1] - points[0])

    arc_length = max(0.0, min(float(arc_length), total_length))
    center_index = int(np.searchsorted(cumulative, arc_length, side="left"))
    prev_index = max(center_index - 1, 0)
    next_index = min(center_index + 1, points.shape[0] - 1)
    tangent = points[next_index] - points[prev_index]
    normalized = normalize_vector(tangent)
    if normalized is not None:
        return normalized

    if center_index < points.shape[0] - 1:
        return normalize_vector(points[center_index + 1] - points[center_index])
    return normalize_vector(points[center_index] - points[center_index - 1])


def smallest_pca_axis(points: np.ndarray) -> Optional[np.ndarray]:
    if points.shape[0] < 3:
        return None
    centered = points - points.mean(axis=0, keepdims=True)
    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular_values.shape[0] < 3:
        return None
    return normalize_vector(vh[-1])


def compute_leaf_width(
    leaf_points: np.ndarray,
    midrib: np.ndarray,
    h_scale: float,
    midrib_length: float,
) -> Tuple[float, int]:
    if midrib.shape[0] < 2 or leaf_points.shape[0] < 20:
        return math.nan, 0

    leaf_tree = cKDTree(leaf_points)
    radius = max(12.0 * h_scale, 0.08 * midrib_length)
    slab_half_thickness = 4.0 * h_scale
    sample_positions = np.linspace(0.15 * midrib_length, 0.85 * midrib_length, WIDTH_SECTION_COUNT)

    widths: List[float] = []
    valid_sections = 0

    for arc_length in sample_positions:
        section_center = sample_polyline_point(midrib, float(arc_length))
        tangent = sample_polyline_tangent(midrib, float(arc_length))
        if tangent is None:
            continue

        neighborhood_indices = leaf_tree.query_ball_point(section_center, r=radius)
        if len(neighborhood_indices) < 3:
            continue

        neighborhood = leaf_points[np.asarray(neighborhood_indices, dtype=np.int32)]
        normal = smallest_pca_axis(neighborhood)
        if normal is None:
            continue

        width_direction = normalize_vector(np.cross(normal, tangent))
        if width_direction is None:
            continue

        centered = neighborhood - section_center
        slab_mask = (
            (np.abs(centered @ tangent) <= slab_half_thickness)
            & (np.abs(centered @ normal) <= slab_half_thickness)
        )
        section_points = neighborhood[slab_mask]
        if section_points.shape[0] < 20:
            continue

        projections = (section_points - section_center) @ width_direction
        widths.append(float(projections.max() - projections.min()))
        valid_sections += 1

    if not widths:
        return math.nan, valid_sections
    return float(max(widths)), valid_sections


def compute_leaf_area(points: np.ndarray, h_scale: float) -> float:
    if points.shape[0] < 3:
        return 0.0

    centered = points - points.mean(axis=0, keepdims=True)
    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    if singular_values.shape[0] < 2:
        return 0.0

    uv = centered @ vh[:2].T
    try:
        triangulation = Delaunay(uv)
    except QhullError:
        return 0.0

    simplices = triangulation.simplices
    if simplices.size == 0:
        return 0.0

    triangles_2d = uv[simplices]
    edge_01 = np.linalg.norm(triangles_2d[:, 0] - triangles_2d[:, 1], axis=1)
    edge_12 = np.linalg.norm(triangles_2d[:, 1] - triangles_2d[:, 2], axis=1)
    edge_20 = np.linalg.norm(triangles_2d[:, 2] - triangles_2d[:, 0], axis=1)
    keep_mask = (
        (edge_01 <= 6.0 * h_scale)
        & (edge_12 <= 6.0 * h_scale)
        & (edge_20 <= 6.0 * h_scale)
    )
    if not np.any(keep_mask):
        return 0.0

    triangles_3d = points[simplices[keep_mask]]
    cross_products = np.cross(
        triangles_3d[:, 1] - triangles_3d[:, 0],
        triangles_3d[:, 2] - triangles_3d[:, 0],
    )
    triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)
    return float(triangle_areas.sum())


def compute_inclination_degrees(midrib: np.ndarray) -> float:
    length = polyline_length(midrib)
    if length <= 0:
        return math.nan

    base_point = sample_polyline_point(midrib, 0.0)
    target_point = sample_polyline_point(midrib, 0.15 * length)
    direction = normalize_vector(target_point - base_point)
    if direction is None:
        return math.nan

    return float(math.degrees(math.asin(abs(float(direction[2])))))


def compute_midrib_midpoint_and_curvature(
    midrib: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    if midrib.shape[0] == 0:
        return np.full((3,), math.nan, dtype=np.float64), math.nan, math.nan

    length = polyline_length(midrib)
    if length <= 0:
        midpoint = np.asarray(midrib[0], dtype=np.float64).copy()
        return midpoint, math.nan, math.nan

    base_point = sample_polyline_point(midrib, 0.0)
    midpoint = sample_polyline_point(midrib, 0.5 * length)
    tip_point = sample_polyline_point(midrib, length)

    midpoint_to_base = normalize_vector(base_point - midpoint)
    midpoint_to_tip = normalize_vector(tip_point - midpoint)
    if midpoint_to_base is None or midpoint_to_tip is None:
        return midpoint, math.nan, math.nan

    cosine_value = float(
        np.clip(np.dot(midpoint_to_base, midpoint_to_tip), -1.0, 1.0)
    )
    midpoint_angle_deg = float(math.degrees(math.acos(cosine_value)))
    curvature_deg = float(180.0 - midpoint_angle_deg)
    return midpoint, midpoint_angle_deg, curvature_deg


def make_invalid_result(
    instance_id: int,
    n_points_raw: int,
    reason: str,
    debug_updates: Dict[str, object],
) -> Tuple[None, Dict[str, object]]:
    debug = {
        "leaf_instance_id": int(instance_id),
        "status": "invalid",
        "reason": reason,
        "n_points_raw": int(n_points_raw),
        "n_points_after_outlier": 0,
        "n_points_used": 0,
        "h_scale": math.nan,
        "tau_stem_initial": math.nan,
        "tau_stem_final": math.nan,
        "seed_count": 0,
        "seed_fallback_used": False,
        "no_contact_seed_fallback_used": False,
        "seed_strategy": "",
        "graph_edge_count": 0,
        "midrib_raw_length": math.nan,
        "valid_width_sections": 0,
        "base_stem_distance": math.nan,
        "tip_stem_distance": math.nan,
    }
    debug.update(debug_updates)
    return None, debug


def process_leaf(
    instance_id: int,
    leaf_frame: pd.DataFrame,
    stem_tree: cKDTree,
    scale_factor: float,
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    raw_points = leaf_frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
    raw_point_indices = leaf_frame["point_index"].to_numpy(dtype=np.int32)
    n_points_raw = int(raw_points.shape[0])

    if n_points_raw < MIN_POINTS_PER_LEAF:
        return make_invalid_result(
            instance_id,
            n_points_raw,
            "raw_points_below_threshold",
            {},
        )

    d8 = mean_neighbor_distance(raw_points, KNN_OUTLIER_NEIGHBORS)
    d8_median = float(np.median(d8))
    d8_mad = median_absolute_deviation(d8)
    outlier_threshold = d8_median + 3.0 * d8_mad
    keep_mask = d8 <= outlier_threshold
    filtered_points = raw_points[keep_mask]
    filtered_point_indices = raw_point_indices[keep_mask]

    if filtered_points.shape[0] < MIN_POINTS_PER_LEAF:
        return make_invalid_result(
            instance_id,
            n_points_raw,
            "points_below_threshold_after_outlier_filter",
            {"n_points_after_outlier": int(filtered_points.shape[0])},
        )

    h_scale = median_second_neighbor_distance(filtered_points)
    if not np.isfinite(h_scale) or h_scale <= 0:
        return make_invalid_result(
            instance_id,
            n_points_raw,
            "invalid_sampling_scale",
            {
                "n_points_after_outlier": int(filtered_points.shape[0]),
                "h_scale": float(h_scale),
            },
        )

    graph = build_knn_graph(filtered_points, GRAPH_NEIGHBORS, 4.0 * h_scale)
    filtered_points, filtered_point_indices, graph = keep_largest_component(
        filtered_points, filtered_point_indices, graph
    )

    if filtered_points.shape[0] < MIN_POINTS_PER_LEAF:
        return make_invalid_result(
            instance_id,
            n_points_raw,
            "points_below_threshold_after_component_filter",
            {
                "n_points_after_outlier": int(keep_mask.sum()),
                "n_points_used": int(filtered_points.shape[0]),
                "h_scale": float(h_scale),
            },
        )

    if graph.nnz == 0:
        return make_invalid_result(
            instance_id,
            n_points_raw,
            "empty_knn_graph",
            {
                "n_points_after_outlier": int(keep_mask.sum()),
                "n_points_used": int(filtered_points.shape[0]),
                "h_scale": float(h_scale),
            },
        )

    d_stem, _ = stem_tree.query(filtered_points, k=1)
    d_stem = np.asarray(d_stem, dtype=np.float64).reshape(-1)
    tau_stem_initial = float(
        min(np.percentile(d_stem, 10.0), 8.0 * h_scale)
    )
    seed_mask = d_stem <= tau_stem_initial
    tau_stem_final = tau_stem_initial
    seed_fallback_used = False
    no_contact_seed_fallback_used = False
    seed_strategy = "stem_threshold_p10"

    if int(seed_mask.sum()) < MIN_STEM_SEED_POINTS:
        tau_stem_final = float(min(np.percentile(d_stem, 20.0), 12.0 * h_scale))
        seed_mask = d_stem <= tau_stem_final
        seed_fallback_used = True
        seed_strategy = "stem_threshold_p20"

    seed_indices = select_seed_cluster(np.flatnonzero(seed_mask), graph, d_stem)
    if seed_indices.size == 0:
        nearest_distance = float(np.min(d_stem))
        tau_no_contact = nearest_distance + NO_CONTACT_DISTANCE_MARGIN * h_scale
        candidate_mask = d_stem <= tau_no_contact
        candidate_indices = np.flatnonzero(candidate_mask)

        if candidate_indices.size < MIN_STEM_SEED_POINTS:
            seed_quota = min(
                filtered_points.shape[0],
                max(
                    MIN_STEM_SEED_POINTS,
                    int(math.ceil(NO_CONTACT_SEED_FRACTION * filtered_points.shape[0])),
                ),
                NO_CONTACT_SEED_MAX_POINTS,
            )
            nearest_indices = np.argsort(d_stem)[:seed_quota]
            candidate_indices = np.unique(
                np.concatenate([candidate_indices, nearest_indices])
            ).astype(np.int32)

        seed_indices = select_seed_cluster(candidate_indices, graph, d_stem)
        tau_stem_final = float(
            max(
                tau_stem_final,
                float(np.max(d_stem[candidate_indices]))
                if candidate_indices.size > 0
                else tau_no_contact,
            )
        )
        no_contact_seed_fallback_used = True
        seed_strategy = "nearest_stem_cluster"

    if seed_indices.size == 0:
        return make_invalid_result(
            instance_id,
            n_points_raw,
            "seed_selection_failed",
            {
                "n_points_after_outlier": int(keep_mask.sum()),
                "n_points_used": int(filtered_points.shape[0]),
                "h_scale": float(h_scale),
                "tau_stem_initial": tau_stem_initial,
                "tau_stem_final": tau_stem_final,
                "seed_fallback_used": seed_fallback_used,
                "no_contact_seed_fallback_used": no_contact_seed_fallback_used,
                "seed_strategy": seed_strategy,
            },
        )

    d_seed = multi_source_shortest_paths(graph, seed_indices)
    finite_seed_mask = np.isfinite(d_seed)
    if not np.any(finite_seed_mask):
        return make_invalid_result(
            instance_id,
            n_points_raw,
            "no_finite_seed_distances",
            {
                "n_points_after_outlier": int(keep_mask.sum()),
                "n_points_used": int(filtered_points.shape[0]),
                "h_scale": float(h_scale),
                "tau_stem_initial": tau_stem_initial,
                "tau_stem_final": tau_stem_final,
                "seed_count": int(seed_indices.size),
                "seed_fallback_used": seed_fallback_used,
                "no_contact_seed_fallback_used": no_contact_seed_fallback_used,
                "seed_strategy": seed_strategy,
                "graph_edge_count": int(graph.nnz // 2),
            },
        )

    tip_local_index = int(np.nanargmax(d_seed))
    d_tip = np.asarray(
        dijkstra(graph, directed=False, indices=tip_local_index), dtype=np.float64
    )
    base_local_index = int(seed_indices[np.nanargmax(d_tip[seed_indices])])

    d_base, predecessors = dijkstra(
        graph,
        directed=False,
        indices=base_local_index,
        return_predecessors=True,
    )
    d_base = np.asarray(d_base, dtype=np.float64)
    predecessors = np.asarray(predecessors, dtype=np.int32)
    if not np.isfinite(d_base[tip_local_index]):
        return make_invalid_result(
            instance_id,
            n_points_raw,
            "base_to_tip_path_missing",
            {
                "n_points_after_outlier": int(keep_mask.sum()),
                "n_points_used": int(filtered_points.shape[0]),
                "h_scale": float(h_scale),
                "tau_stem_initial": tau_stem_initial,
                "tau_stem_final": tau_stem_final,
                "seed_count": int(seed_indices.size),
                "seed_fallback_used": seed_fallback_used,
                "no_contact_seed_fallback_used": no_contact_seed_fallback_used,
                "seed_strategy": seed_strategy,
                "graph_edge_count": int(graph.nnz // 2),
            },
        )

    try:
        path_local_indices = reconstruct_path(
            predecessors, base_local_index, tip_local_index
        )
    except ValueError:
        return make_invalid_result(
            instance_id,
            n_points_raw,
            "path_reconstruction_failed",
            {
                "n_points_after_outlier": int(keep_mask.sum()),
                "n_points_used": int(filtered_points.shape[0]),
                "h_scale": float(h_scale),
                "tau_stem_initial": tau_stem_initial,
                "tau_stem_final": tau_stem_final,
                "seed_count": int(seed_indices.size),
                "seed_fallback_used": seed_fallback_used,
                "graph_edge_count": int(graph.nnz // 2),
            },
        )

    raw_midrib = filtered_points[path_local_indices]
    raw_midrib_length = polyline_length(raw_midrib)
    if raw_midrib.shape[0] < 2 or raw_midrib_length <= 0:
        return make_invalid_result(
            instance_id,
            n_points_raw,
            "invalid_raw_midrib",
            {
                "n_points_after_outlier": int(keep_mask.sum()),
                "n_points_used": int(filtered_points.shape[0]),
                "h_scale": float(h_scale),
                "tau_stem_initial": tau_stem_initial,
                "tau_stem_final": tau_stem_final,
                "seed_count": int(seed_indices.size),
                "seed_fallback_used": seed_fallback_used,
                "no_contact_seed_fallback_used": no_contact_seed_fallback_used,
                "seed_strategy": seed_strategy,
                "graph_edge_count": int(graph.nnz // 2),
                "midrib_raw_length": float(raw_midrib_length),
            },
        )

    resample_step = max(3.0 * h_scale, 0.02 * raw_midrib_length)
    midrib = smooth_polyline(resample_polyline(raw_midrib, resample_step))
    midrib_length = polyline_length(midrib)
    midpoint, midpoint_angle_deg, curvature_deg = (
        compute_midrib_midpoint_and_curvature(midrib)
    )
    inclination_deg = compute_inclination_degrees(midrib)
    max_leaf_width, valid_width_sections = compute_leaf_width(
        filtered_points, midrib, h_scale, midrib_length
    )
    area_3d = compute_leaf_area(filtered_points, h_scale)

    base_stem_distance = float(d_stem[base_local_index])
    tip_stem_distance = float(d_stem[tip_local_index])
    if not base_stem_distance < tip_stem_distance:
        return make_invalid_result(
            instance_id,
            n_points_raw,
            "base_not_closer_to_stem_than_tip",
            {
                "n_points_after_outlier": int(keep_mask.sum()),
                "n_points_used": int(filtered_points.shape[0]),
                "h_scale": float(h_scale),
                "tau_stem_initial": tau_stem_initial,
                "tau_stem_final": tau_stem_final,
                "seed_count": int(seed_indices.size),
                "seed_fallback_used": seed_fallback_used,
                "no_contact_seed_fallback_used": no_contact_seed_fallback_used,
                "seed_strategy": seed_strategy,
                "graph_edge_count": int(graph.nnz // 2),
                "midrib_raw_length": float(raw_midrib_length),
                "valid_width_sections": int(valid_width_sections),
                "base_stem_distance": base_stem_distance,
                "tip_stem_distance": tip_stem_distance,
            },
        )

    result = {
        "leaf_instance_id": int(instance_id),
        "n_points_raw": int(n_points_raw),
        "n_points_used": int(filtered_points.shape[0]),
        "base_point_index": int(filtered_point_indices[base_local_index]),
        "tip_point_index": int(filtered_point_indices[tip_local_index]),
        "base_x": float(filtered_points[base_local_index, 0]),
        "base_y": float(filtered_points[base_local_index, 1]),
        "base_z": float(filtered_points[base_local_index, 2]),
        "tip_x": float(filtered_points[tip_local_index, 0]),
        "tip_y": float(filtered_points[tip_local_index, 1]),
        "tip_z": float(filtered_points[tip_local_index, 2]),
        "midpoint_x": float(midpoint[0]),
        "midpoint_y": float(midpoint[1]),
        "midpoint_z": float(midpoint[2]),
        "midrib_length": float(midrib_length),
        "max_leaf_width": (
            float(max_leaf_width) if np.isfinite(max_leaf_width) else math.nan
        ),
        "midpoint_angle_deg": (
            float(midpoint_angle_deg)
            if np.isfinite(midpoint_angle_deg)
            else math.nan
        ),
        "curvature_deg": (
            float(curvature_deg) if np.isfinite(curvature_deg) else math.nan
        ),
        "inclination_deg": (
            float(inclination_deg) if np.isfinite(inclination_deg) else math.nan
        ),
        "area_3d": float(area_3d),
        "scale_factor": float(scale_factor),
    }
    debug = {
        "leaf_instance_id": int(instance_id),
        "status": "valid",
        "reason": "",
        "n_points_raw": int(n_points_raw),
        "n_points_after_outlier": int(keep_mask.sum()),
        "n_points_used": int(filtered_points.shape[0]),
        "h_scale": float(h_scale),
        "tau_stem_initial": tau_stem_initial,
        "tau_stem_final": tau_stem_final,
        "seed_count": int(seed_indices.size),
        "seed_fallback_used": bool(seed_fallback_used),
        "no_contact_seed_fallback_used": bool(no_contact_seed_fallback_used),
        "seed_strategy": seed_strategy,
        "graph_edge_count": int(graph.nnz // 2),
        "midrib_raw_length": float(raw_midrib_length),
        "valid_width_sections": int(valid_width_sections),
        "base_stem_distance": base_stem_distance,
        "tip_stem_distance": tip_stem_distance,
    }
    return result, debug


def write_outputs(
    output_dir: Path,
    leaf_results: Sequence[Dict[str, object]],
    debug_rows: Sequence[Dict[str, object]],
    stem_point_count: int,
    scale_factor: float,
    summary_metadata: Optional[Dict[str, object]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    leaf_columns = [
        "leaf_instance_id",
        "n_points_raw",
        "n_points_used",
        "base_point_index",
        "tip_point_index",
        "base_x",
        "base_y",
        "base_z",
        "tip_x",
        "tip_y",
        "tip_z",
        "midpoint_x",
        "midpoint_y",
        "midpoint_z",
        "midrib_length",
        "max_leaf_width",
        "midpoint_angle_deg",
        "curvature_deg",
        "inclination_deg",
        "area_3d",
        "scale_factor",
    ]
    leaf_frame = pd.DataFrame(list(leaf_results), columns=leaf_columns)
    if not leaf_frame.empty:
        leaf_frame = leaf_frame.sort_values("leaf_instance_id").reset_index(drop=True)
    leaf_frame.to_csv(output_dir / "leaf_traits.csv", index=False)

    debug_frame = pd.DataFrame(list(debug_rows))
    if not debug_frame.empty and "leaf_instance_id" in debug_frame.columns:
        debug_frame = debug_frame.sort_values("leaf_instance_id").reset_index(drop=True)
    debug_frame.to_csv(output_dir / "leaf_debug.csv", index=False)

    if leaf_frame.empty:
        summary = {
            "leaf_count": 0,
            "stem_point_count": int(stem_point_count),
            "total_leaf_area_3d": 0.0,
            "mean_leaf_length": math.nan,
            "mean_leaf_width": math.nan,
            "mean_leaf_curvature_deg": math.nan,
            "mean_inclination_deg": math.nan,
            "max_leaf_length": math.nan,
            "max_leaf_width": math.nan,
            "max_leaf_curvature_deg": math.nan,
            "scale_factor": float(scale_factor),
        }
    else:
        summary = {
            "leaf_count": int(len(leaf_frame)),
            "stem_point_count": int(stem_point_count),
            "total_leaf_area_3d": float(leaf_frame["area_3d"].sum()),
            "mean_leaf_length": float(leaf_frame["midrib_length"].mean()),
            "mean_leaf_width": float(leaf_frame["max_leaf_width"].mean(skipna=True)),
            "mean_leaf_curvature_deg": float(
                leaf_frame["curvature_deg"].mean(skipna=True)
            ),
            "mean_inclination_deg": float(
                leaf_frame["inclination_deg"].mean(skipna=True)
            ),
            "max_leaf_length": float(leaf_frame["midrib_length"].max()),
            "max_leaf_width": float(leaf_frame["max_leaf_width"].max(skipna=True)),
            "max_leaf_curvature_deg": float(
                leaf_frame["curvature_deg"].max(skipna=True)
            ),
            "scale_factor": float(scale_factor),
        }

    if summary_metadata:
        summary.update(summary_metadata)

    with (output_dir / "plant_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    frame = load_prediction_ply(input_path, scale=args.scale)
    leaf_frame = frame[
        (frame["semantic_id"] == 1) & (frame["instance_id"] > 0)
    ].copy()
    if leaf_frame.empty:
        raise SystemExit("No valid leaf points found in pred_point_labels.ply.")

    stem_frame = frame[frame["semantic_id"] == 0].copy()
    stem_point_count = int(stem_frame.shape[0])
    summary_metadata: Dict[str, object]
    if stem_frame.empty:
        stem_points, summary_metadata = estimate_pseudo_stem_points(leaf_frame)
    else:
        stem_points = stem_frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
        summary_metadata = {
            "stem_source": "semantic_stem",
            "estimated_stem_center": None,
            "estimated_stem_point_count": int(stem_points.shape[0]),
            "estimated_stem_method": None,
        }

    stem_tree = cKDTree(stem_points)

    leaf_results: List[Dict[str, object]] = []
    debug_rows: List[Dict[str, object]] = []

    for instance_id, group in leaf_frame.groupby("instance_id", sort=True):
        result, debug = process_leaf(
            int(instance_id), group.copy(), stem_tree, float(args.scale)
        )
        debug_rows.append(debug)
        if result is not None:
            leaf_results.append(result)

    write_outputs(
        output_dir=output_dir,
        leaf_results=leaf_results,
        debug_rows=debug_rows,
        stem_point_count=stem_point_count,
        scale_factor=float(args.scale),
        summary_metadata=summary_metadata,
    )

    print(f"Processed {len(debug_rows)} leaf instances.")
    print(f"Valid leaves: {len(leaf_results)}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
