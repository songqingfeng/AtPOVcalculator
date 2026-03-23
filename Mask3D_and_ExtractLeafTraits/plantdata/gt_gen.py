import os
import argparse
import numpy as np
import yaml


def main(root="data/processed/arabidopsis", mode="validation", semantic_offset=0):
    with open(os.path.join(root, f"{mode}_database.yaml")) as db_file:
        db = yaml.safe_load(db_file)

    gt_root = os.path.join(root, "instance_gt", mode)
    os.makedirs(gt_root, exist_ok=True)

    print(
        "Generating GT files "
        f"(mode={mode}, semantic_offset={semantic_offset}, encoding=semantic*1000+instance)..."
    )
    for record in db:
        pts = np.load(record["filepath"])
        sem = pts[:, 10].astype(np.int32)
        inst = pts[:, 11].astype(np.int32)

        # 0 means "no instance" in evaluation utilities; enforce positive ids.
        if np.any(inst <= 0):
            raise ValueError(
                f"scene={record['scene']} has non-positive instance ids. "
                "Please ensure all points have instance_id > 0."
            )
        # benchmark encoding requires instance_id < 1000.
        if np.any(inst >= 1000):
            raise ValueError(
                f"scene={record['scene']} has instance_id >= 1000, incompatible with semantic*1000+instance encoding."
            )

        gt_sem = sem + int(semantic_offset)
        if np.any(gt_sem < 0):
            raise ValueError(
                f"scene={record['scene']} has negative semantic ids after semantic_offset={semantic_offset}."
            )

        gt = gt_sem * 1000 + inst
        np.savetxt(os.path.join(gt_root, record["scene"] + ".txt"), gt, fmt="%d")

    print(f"GT generation complete: {gt_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/processed/arabidopsis")
    parser.add_argument("--mode", default="validation", choices=["train", "validation"])
    parser.add_argument(
        "--semantic-offset",
        type=int,
        default=0,
        help="Add this offset to semantic ids before GT export. "
        "Use 1 to avoid semantic prefix 0 in GT.",
    )
    args = parser.parse_args()
    main(root=args.root, mode=args.mode, semantic_offset=args.semantic_offset)
