import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm


def _is_integer_like(values, atol=1e-6):
    if values.size == 0:
        return False
    return np.all(np.abs(values - np.round(values)) <= atol)


def _select_instance_col(data):
    n_cols = data.shape[1]
    # Candidate columns: from column 7 to the 4th-to-last column.
    candidate_cols = list(range(6, n_cols - 3))
    if not candidate_cols:
        raise ValueError(
            f"Not enough columns ({n_cols}) to search instance id from column 7 onward."
        )

    candidates = []
    for col_idx in candidate_cols:
        col = data[:, col_idx]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        if not _is_integer_like(finite):
            continue

        rounded = np.round(finite).astype(np.int64)
        unique_vals = np.unique(rounded)
        if unique_vals.size <= 1:
            continue

        nan_count = int(np.isnan(col).sum())
        nan_ratio = float(nan_count / float(col.shape[0]))
        candidates.append(
            {
                "col_idx": col_idx,
                "all_finite": bool(np.isfinite(col).all()),
                "has_zero": bool(np.any(unique_vals == 0)),
                "has_positive": bool(np.any(unique_vals > 0)),
                "unique_count": int(unique_vals.size),
                "max_value": int(unique_vals.max()),
                "nan_count": nan_count,
                "nan_ratio": nan_ratio,
            }
        )

    if not candidates:
        raise ValueError(
            "No integer-like instance column found between column 7 and the 4th-to-last column."
        )

    # New rule:
    # 1) Prefer columns containing NaN (NaN marks stem points).
    # 2) If multiple, choose the one with lower NaN ratio.
    # 3) Tie-break with richer id range.
    nan_candidates = [c for c in candidates if c["nan_count"] > 0 and c["has_positive"]]
    if nan_candidates:
        best = min(
            nan_candidates,
            key=lambda c: (
                c["nan_ratio"],
                -c["unique_count"],
                -c["max_value"],
                c["col_idx"],
            ),
        )
        return best["col_idx"]

    # Fallback: no NaN column found, pick the best finite integer-like id column.
    pool = [c for c in candidates if c["has_positive"]] or candidates
    best = max(pool, key=lambda c: (c["unique_count"], c["max_value"], -c["col_idx"]))
    return best["col_idx"]


def _remap_leaf_instance_ids(raw_leaf_ids, reserved_ids, max_instance_id=999):
    mapping = {}
    used = set(int(x) for x in reserved_ids)
    next_auto = 1
    if used:
        next_auto = max(used) + 1

    for old_id in sorted(np.unique(raw_leaf_ids).astype(np.int64).tolist()):
        preferred = int(old_id)
        if (
            preferred > 0
            and preferred <= max_instance_id
            and preferred not in used
        ):
            new_id = preferred
        else:
            while next_auto in used and next_auto <= max_instance_id:
                next_auto += 1
            if next_auto > max_instance_id:
                raise ValueError(
                    f"instance id overflow: need <= {max_instance_id}, used={len(used)}"
                )
            new_id = next_auto
            next_auto += 1
        mapping[preferred] = int(new_id)
        used.add(int(new_id))
    return mapping


def process_single_file(txt_path, save_dir):
    try:
        data = np.loadtxt(txt_path)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < 9:
            raise ValueError(
                f"Expected at least 9 columns (xyz/rgb/normals), got {data.shape[1]}"
            )

        xyz = data[:, 0:3].astype(np.float32)
        rgb = data[:, 3:6].astype(np.float32) / 255.0
        # The last 3 columns are treated as normals for arabidopsis files.
        normals = np.nan_to_num(data[:, -3:], nan=0.0).astype(np.float32)
        inst_col = _select_instance_col(data)
        inst_source = data[:, inst_col]
        stem_mask = np.isnan(inst_source)
        leaf_mask = ~stem_mask

        # If inf/-inf appears, treat as stem-like unknowns to avoid invalid ids.
        inf_mask = np.isinf(inst_source)
        if np.any(inf_mask):
            stem_mask = stem_mask | inf_mask
            leaf_mask = ~stem_mask

        xyz -= xyz.min(axis=0)
        xyz *= 100.0

        # No background:
        # semantic 0 -> Stem, semantic 1 -> Leaf
        sem = np.zeros((xyz.shape[0], 1), dtype=np.int32)
        sem[leaf_mask] = 1

        # Instance ids:
        # - all stem points are one instance
        # - leaf points preserve original ids as much as possible without conflicts
        # - keep ids > 0 (0 is treated as "no instance" by evaluation utils)
        stem_instance_id = 1
        inst_out = np.zeros(xyz.shape[0], dtype=np.int32)
        inst_out[stem_mask] = stem_instance_id

        if np.any(leaf_mask):
            leaf_inst_ids = np.round(inst_source[leaf_mask]).astype(np.int64)
            inst_mapping = _remap_leaf_instance_ids(
                raw_leaf_ids=leaf_inst_ids, reserved_ids={stem_instance_id}
            )
            rounded_leaf_full = np.zeros(xyz.shape[0], dtype=np.int64)
            rounded_leaf_full[leaf_mask] = leaf_inst_ids
            for old_id, new_id in inst_mapping.items():
                inst_out[leaf_mask & (rounded_leaf_full == old_id)] = new_id

        segments = inst_out.copy()

        out_data = np.hstack(
            [
                xyz,
                rgb,
                normals,
                segments[:, None].astype(np.float32),
                sem.astype(np.float32),
                inst_out[:, None].astype(np.float32),
            ]
        )

        save_path = save_dir / (txt_path.stem + ".npy")
        np.save(save_path, out_data.astype(np.float32))
        return {
            "filepath": str(save_path.absolute()),
            "scene": txt_path.stem,
            "file_len": xyz.shape[0],
            "color_mean": [0.5, 0.5, 0.5],
            "color_std": [0.5, 0.5, 0.5],
        }
    except Exception as e:
        print(f"Error processing {txt_path}: {e}")
        return None


def main():
    save_root = Path("data/processed/arabidopsis")
    input_folder = Path(".")
    txt_files = sorted(input_folder.glob("*.txt"))

    for mode in ["train", "validation"]:
        (save_root / mode).mkdir(parents=True, exist_ok=True)

    split_idx = int(len(txt_files) * 0.8)

    db_train = []
    for txt_file in tqdm(txt_files[:split_idx], desc="Train"):
        result = process_single_file(txt_file, save_root / "train")
        if result:
            db_train.append(result)

    db_val = []
    for txt_file in tqdm(txt_files[split_idx:], desc="Val"):
        result = process_single_file(txt_file, save_root / "validation")
        if result:
            db_val.append(result)

    with open(save_root / "train_database.yaml", "w") as train_file:
        yaml.dump(db_train, train_file)
    with open(save_root / "validation_database.yaml", "w") as val_file:
        yaml.dump(db_val, val_file)

    labels = {
        0: {"color": [165, 42, 42], "name": "Stem", "validation": True},
        1: {"color": [0, 255, 0], "name": "Leaf", "validation": True},
    }
    with open(save_root / "label_database.yaml", "w") as label_file:
        yaml.dump(labels, label_file)


if __name__ == "__main__":
    main()
