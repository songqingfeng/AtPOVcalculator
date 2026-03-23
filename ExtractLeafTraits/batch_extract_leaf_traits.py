from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import extract_leaf_traits_from_ply as traits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch extract leaf traits for all plant folders under a parent "
            "directory. Each plant folder must contain pred_point_labels.ply."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Parent directory containing one subdirectory per plant.",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="pred_point_labels.ply",
        help="PLY filename expected inside each plant subdirectory.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Coordinate scaling factor applied before geometry extraction.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="longest_leaf_summary.csv",
        help="Filename for the parent-level summary CSV.",
    )
    return parser.parse_args()


def find_plant_dirs(root_dir: Path, input_name: str) -> List[Path]:
    return sorted(
        path for path in root_dir.iterdir() if path.is_dir() and (path / input_name).exists()
    )


def prepare_stem_tree(
    frame: pd.DataFrame, leaf_frame: pd.DataFrame
) -> Tuple[cKDTree, int, Dict[str, object]]:
    stem_frame = frame[frame["semantic_id"] == 0].copy()
    stem_point_count = int(stem_frame.shape[0])
    if stem_frame.empty:
        stem_points, summary_metadata = traits.estimate_pseudo_stem_points(leaf_frame)
    else:
        stem_points = stem_frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
        summary_metadata = {
            "stem_source": "semantic_stem",
            "estimated_stem_center": None,
            "estimated_stem_point_count": int(stem_points.shape[0]),
            "estimated_stem_method": None,
        }
    return cKDTree(stem_points), stem_point_count, summary_metadata


def build_empty_summary_row(
    plant_id: str,
    plant_dir: Path,
    input_path: Path,
    status: str,
    error_message: str = "",
) -> Dict[str, object]:
    return {
        "plant_id": plant_id,
        "status": status,
        "error_message": error_message,
        "input_path": str(input_path),
        "plant_dir": str(plant_dir),
        "valid_leaf_count": 0,
        "stem_source": "",
        "stem_point_count": 0,
        "estimated_stem_point_count": 0,
        "longest_leaf_instance_id": math.nan,
        "longest_n_points_raw": math.nan,
        "longest_n_points_used": math.nan,
        "longest_base_point_index": math.nan,
        "longest_tip_point_index": math.nan,
        "longest_base_x": math.nan,
        "longest_base_y": math.nan,
        "longest_base_z": math.nan,
        "longest_tip_x": math.nan,
        "longest_tip_y": math.nan,
        "longest_tip_z": math.nan,
        "longest_midpoint_x": math.nan,
        "longest_midpoint_y": math.nan,
        "longest_midpoint_z": math.nan,
        "longest_midrib_length": math.nan,
        "longest_max_leaf_width": math.nan,
        "longest_midpoint_angle_deg": math.nan,
        "longest_curvature_deg": math.nan,
        "longest_inclination_deg": math.nan,
        "longest_area_3d": math.nan,
        "scale_factor": math.nan,
    }


def build_longest_leaf_row(
    plant_id: str,
    plant_dir: Path,
    input_path: Path,
    leaf_results: Sequence[Dict[str, object]],
    stem_point_count: int,
    summary_metadata: Dict[str, object],
) -> Dict[str, object]:
    leaf_frame = pd.DataFrame(list(leaf_results))
    longest_index = int(leaf_frame["midrib_length"].idxmax())
    longest_leaf = leaf_frame.loc[longest_index]
    return {
        "plant_id": plant_id,
        "status": "ok",
        "error_message": "",
        "input_path": str(input_path),
        "plant_dir": str(plant_dir),
        "valid_leaf_count": int(len(leaf_frame)),
        "stem_source": str(summary_metadata.get("stem_source", "")),
        "stem_point_count": int(stem_point_count),
        "estimated_stem_point_count": int(
            summary_metadata.get("estimated_stem_point_count", 0)
        ),
        "longest_leaf_instance_id": int(longest_leaf["leaf_instance_id"]),
        "longest_n_points_raw": int(longest_leaf["n_points_raw"]),
        "longest_n_points_used": int(longest_leaf["n_points_used"]),
        "longest_base_point_index": int(longest_leaf["base_point_index"]),
        "longest_tip_point_index": int(longest_leaf["tip_point_index"]),
        "longest_base_x": float(longest_leaf["base_x"]),
        "longest_base_y": float(longest_leaf["base_y"]),
        "longest_base_z": float(longest_leaf["base_z"]),
        "longest_tip_x": float(longest_leaf["tip_x"]),
        "longest_tip_y": float(longest_leaf["tip_y"]),
        "longest_tip_z": float(longest_leaf["tip_z"]),
        "longest_midpoint_x": float(longest_leaf["midpoint_x"]),
        "longest_midpoint_y": float(longest_leaf["midpoint_y"]),
        "longest_midpoint_z": float(longest_leaf["midpoint_z"]),
        "longest_midrib_length": float(longest_leaf["midrib_length"]),
        "longest_max_leaf_width": float(longest_leaf["max_leaf_width"]),
        "longest_midpoint_angle_deg": float(longest_leaf["midpoint_angle_deg"]),
        "longest_curvature_deg": float(longest_leaf["curvature_deg"]),
        "longest_inclination_deg": float(longest_leaf["inclination_deg"]),
        "longest_area_3d": float(longest_leaf["area_3d"]),
        "scale_factor": float(longest_leaf["scale_factor"]),
    }


def process_plant_dir(
    plant_dir: Path,
    input_name: str,
    scale: float,
) -> Dict[str, object]:
    plant_id = plant_dir.name
    input_path = plant_dir / input_name

    frame = traits.load_prediction_ply(input_path, scale=scale)
    leaf_frame = frame[
        (frame["semantic_id"] == 1) & (frame["instance_id"] > 0)
    ].copy()

    if leaf_frame.empty:
        traits.write_outputs(
            output_dir=plant_dir,
            leaf_results=[],
            debug_rows=[],
            stem_point_count=0,
            scale_factor=float(scale),
            summary_metadata={
                "stem_source": "unavailable",
                "estimated_stem_center": None,
                "estimated_stem_point_count": 0,
                "estimated_stem_method": None,
            },
        )
        return build_empty_summary_row(
            plant_id=plant_id,
            plant_dir=plant_dir,
            input_path=input_path,
            status="no_valid_leaf_points",
        )

    stem_tree, stem_point_count, summary_metadata = prepare_stem_tree(frame, leaf_frame)

    leaf_results: List[Dict[str, object]] = []
    debug_rows: List[Dict[str, object]] = []
    for instance_id, group in leaf_frame.groupby("instance_id", sort=True):
        result, debug = traits.process_leaf(
            int(instance_id), group.copy(), stem_tree, float(scale)
        )
        debug_rows.append(debug)
        if result is not None:
            leaf_results.append(result)

    traits.write_outputs(
        output_dir=plant_dir,
        leaf_results=leaf_results,
        debug_rows=debug_rows,
        stem_point_count=stem_point_count,
        scale_factor=float(scale),
        summary_metadata=summary_metadata,
    )

    if not leaf_results:
        row = build_empty_summary_row(
            plant_id=plant_id,
            plant_dir=plant_dir,
            input_path=input_path,
            status="no_valid_leaves_after_filtering",
        )
        row["stem_source"] = str(summary_metadata.get("stem_source", ""))
        row["stem_point_count"] = int(stem_point_count)
        row["estimated_stem_point_count"] = int(
            summary_metadata.get("estimated_stem_point_count", 0)
        )
        row["scale_factor"] = float(scale)
        return row

    return build_longest_leaf_row(
        plant_id=plant_id,
        plant_dir=plant_dir,
        input_path=input_path,
        leaf_results=leaf_results,
        stem_point_count=stem_point_count,
        summary_metadata=summary_metadata,
    )


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    if not root_dir.exists():
        raise SystemExit(f"Root directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise SystemExit(f"Root path is not a directory: {root_dir}")

    plant_dirs = find_plant_dirs(root_dir, args.input_name)
    if not plant_dirs:
        raise SystemExit(
            f"No plant subdirectories containing {args.input_name} were found under {root_dir}"
        )

    summary_rows: List[Dict[str, object]] = []
    for plant_dir in plant_dirs:
        try:
            row = process_plant_dir(
                plant_dir=plant_dir,
                input_name=args.input_name,
                scale=float(args.scale),
            )
        except Exception as exc:
            row = build_empty_summary_row(
                plant_id=plant_dir.name,
                plant_dir=plant_dir,
                input_path=plant_dir / args.input_name,
                status="error",
                error_message=str(exc),
            )
        summary_rows.append(row)
        print(f"{plant_dir.name}: {row['status']}")

    summary_frame = pd.DataFrame(summary_rows)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values("plant_id").reset_index(drop=True)
    summary_path = root_dir / args.summary_name
    summary_frame.to_csv(summary_path, index=False)

    print(f"Processed plant folders: {len(summary_rows)}")
    print(f"Parent summary written to: {summary_path}")


if __name__ == "__main__":
    main()
