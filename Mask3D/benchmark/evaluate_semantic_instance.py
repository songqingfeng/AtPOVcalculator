import math
import os, sys, argparse
from pathlib import Path
import numpy as np
from scipy import stats
from copy import deepcopy
from collections import defaultdict
from uuid import uuid4
import yaml
import benchmark.util as util
import benchmark.util_3d as util_3d

# 全局变量占位，将被 evaluate 函数覆盖
CLASS_LABELS = []
VALID_CLASS_IDS = []
ID_TO_LABEL = {}
LABEL_TO_ID = {}
opt = {
    "overlaps": np.append(np.arange(0.5, 0.95, 0.05), 0.25),
    "min_region_sizes": np.array([100]),
    "distance_threshes": np.array([float("inf")]),
    "distance_confs": np.array([-float("inf")])
}


def _safe_class_name(name, class_id):
    raw = str(name).strip().lower().replace(" ", "_")
    if raw == "":
        raw = f"class_{class_id}"
    return raw


def _configure_arabidopsis_classes_from_label_db(gt_path):
    # gt_path is usually: <data_root>/instance_gt/<mode>
    # Try to find <data_root>/label_database.yaml
    gt_dir = Path(gt_path)
    data_root = gt_dir.parent.parent
    label_db = data_root / "label_database.yaml"

    default_labels = ["leaf"]
    default_ids = np.array([1], dtype=np.int32)
    default_id_to_label = {1: "leaf"}
    default_label_to_id = {"leaf": 1}

    if not label_db.exists():
        return default_labels, default_ids, default_id_to_label, default_label_to_id

    try:
        with open(label_db, "r", encoding="utf-8") as f:
            db = yaml.safe_load(f) or {}
    except Exception:
        return default_labels, default_ids, default_id_to_label, default_label_to_id

    if not isinstance(db, dict):
        return default_labels, default_ids, default_id_to_label, default_label_to_id

    selected = []
    for k, v in db.items():
        try:
            class_id = int(k)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        if not bool(v.get("validation", True)):
            continue
        class_name = _safe_class_name(v.get("name", f"class_{class_id}"), class_id)
        selected.append((class_id, class_name))

    if not selected:
        return default_labels, default_ids, default_id_to_label, default_label_to_id

    selected = sorted(selected, key=lambda x: x[0])
    used_names = set()
    class_labels = []
    valid_class_ids = []
    for class_id, class_name in selected:
        name = class_name
        if name in used_names:
            name = f"{name}_{class_id}"
        used_names.add(name)
        class_labels.append(name)
        valid_class_ids.append(class_id)

    valid_class_ids = np.array(valid_class_ids, dtype=np.int32)
    id_to_label = {cid: cname for cid, cname in zip(valid_class_ids, class_labels)}
    label_to_id = {cname: cid for cid, cname in zip(valid_class_ids, class_labels)}
    return class_labels, valid_class_ids, id_to_label, label_to_id


def _validate_prediction_shapes(pred, gt_ids, scan_key):
    required_keys = ("pred_classes", "pred_scores", "pred_masks")
    for key in required_keys:
        if key not in pred:
            raise KeyError(f"missing key '{key}' in prediction for scan '{scan_key}'")

    pred_masks = np.asarray(pred["pred_masks"])
    if pred_masks.ndim != 2:
        raise IndexError(
            f"pred_masks must be 2D, got shape={pred_masks.shape} for scan '{scan_key}'"
        )

    n_classes = len(pred["pred_classes"])
    n_scores = len(pred["pred_scores"])
    n_masks = pred_masks.shape[1]
    if not (n_classes == n_scores == n_masks):
        raise IndexError(
            f"shape mismatch in scan '{scan_key}': "
            f"n_classes={n_classes}, n_scores={n_scores}, n_masks={n_masks}, masks_shape={pred_masks.shape}"
        )

    n_points = pred_masks.shape[0]
    if n_points != len(gt_ids):
        raise IndexError(
            f"point count mismatch in scan '{scan_key}': pred_points={n_points}, gt_points={len(gt_ids)}"
        )
    return pred_masks

def evaluate_matches(matches):
    overlaps = opt["overlaps"]
    min_region_sizes = [opt["min_region_sizes"][0]]
    dist_threshes = [opt["distance_threshes"][0]]
    dist_confs = [opt["distance_confs"][0]]

    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)
    
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for label_name in CLASS_LABELS:
                    for p in matches[m]["pred"][label_name]:
                        if "uuid" in p:
                            pred_visited[p["uuid"]] = False
            
            for li, label_name in enumerate(CLASS_LABELS):
                y_true, y_score = np.empty(0), np.empty(0)
                hard_false_negatives = 0
                has_gt, has_pred = False, False
                
                for m in matches:
                    pred_instances = matches[m]["pred"][label_name]
                    gt_instances = matches[m]["gt"][label_name]
                    gt_instances = [gt for gt in gt_instances if gt["vert_count"] >= min_region_size]
                    
                    if gt_instances: has_gt = True
                    if pred_instances: has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        for pred in gt["matched_pred"]:
                            if pred_visited[pred["uuid"]]: continue
                            overlap = float(pred["intersection"]) / (gt["vert_count"] + pred["vert_count"] - pred["intersection"])
                            if overlap > overlap_th:
                                confidence = pred["confidence"]
                                if cur_match[gti]:
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min(cur_score[gti], confidence))
                                    cur_score[gti] = max(cur_score[gti], confidence)
                                    cur_match = np.append(cur_match, True)
                                    pred_visited[pred["uuid"]] = True
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred["uuid"]] = True
                        if not found_match: hard_false_negatives += 1
                    
                    y_true = np.append(y_true, cur_true[cur_match])
                    y_score = np.append(y_score, cur_score[cur_match])

                    for pred in pred_instances:
                        if not any(float(gt["intersection"]) / (gt["vert_count"] + pred["vert_count"] - gt["intersection"]) > overlap_th for gt in pred["matched_gt"]):
                            y_true = np.append(y_true, 0)
                            y_score = np.append(y_score, pred["confidence"])

                if has_gt and has_pred:
                    score_arg_sort = np.argsort(y_score)[::-1]
                    y_true_sorted = y_true[score_arg_sort]
                    y_score_sorted = y_score[score_arg_sort]
                    
                    tp = (y_true_sorted == 1).cumsum()
                    fp = (y_true_sorted == 0).cumsum()
                    n_gt = tp[-1] + hard_false_negatives if len(tp)>0 else hard_false_negatives
                    
                    rec = tp / n_gt if n_gt > 0 else np.zeros_like(tp)
                    prec = tp / (tp + fp)
                    
                    ap_current = 0
                    for i in range(11):
                        t = i / 10.0
                        p = prec[rec >= t].max() if any(rec >= t) else 0
                        ap_current += p / 11.0
                else:
                    ap_current = 0.0 if has_gt else float("nan")
                ap[di, li, oi] = ap_current
    return ap

def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(opt["overlaps"], 0.5))[0]
    o25 = np.where(np.isclose(opt["overlaps"], 0.25))[0]
    
    avg_dict = {
        "all_ap": np.nanmean(aps[d_inf, :, :]), 
        "all_ap_50%": np.nanmean(aps[d_inf, :, o50]),
        "all_ap_25%": np.nanmean(aps[d_inf, :, o25]),
        "classes": {}
    }
    
    for li, label_name in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name] = {
            "ap": np.mean(aps[d_inf, li, :]), 
            "ap50%": np.mean(aps[d_inf, li, o50]),
            "ap25%": np.mean(aps[d_inf, li, o25])
        }
    return avg_dict


def print_results(avgs):
    sep = ""
    col1 = ":"
    line_len = 64

    print("")
    print("#" * line_len)
    line = "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("AP") + sep
    line += "{:>15}".format("AP_50%") + sep
    line += "{:>15}".format("AP_25%") + sep
    print(line)
    print("#" * line_len)

    for label_name in CLASS_LABELS:
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50 = avgs["classes"][label_name]["ap50%"]
        ap_25 = avgs["classes"][label_name]["ap25%"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg) + sep
        line += sep + "{:>15.3f}".format(ap_50) + sep
        line += sep + "{:>15.3f}".format(ap_25) + sep
        print(line)

    print("-" * line_len)
    line = "{:<15}".format("average") + sep + col1
    line += sep + "{:>15.3f}".format(avgs["all_ap"]) + sep
    line += sep + "{:>15.3f}".format(avgs["all_ap_50%"]) + sep
    line += sep + "{:>15.3f}".format(avgs["all_ap_25%"]) + sep
    print(line)
    print("")


def write_result_file(avgs, output_file):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("class,class id,ap,ap50,ap25\n")
        for i, label_name in enumerate(CLASS_LABELS):
            class_id = VALID_CLASS_IDS[i]
            ap = avgs["classes"][label_name]["ap"]
            ap50 = avgs["classes"][label_name]["ap50%"]
            ap25 = avgs["classes"][label_name]["ap25%"]
            f.write(f"{label_name},{class_id},{ap:.3f},{ap50:.3f},{ap25:.3f}\n")

def assign_instances_for_scan(pred, gt_file, scan_key="<unknown>"):
    gt_ids = util_3d.load_ids(gt_file)
    pred_masks = _validate_prediction_shapes(pred, gt_ids, scan_key)
    gt_instances = util_3d.get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)
    
    pred2gt = {label: [] for label in CLASS_LABELS}
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]: gt["matched_pred"] = []

    for i in range(len(pred["pred_classes"])):
        label_id = int(pred["pred_classes"][i])
        if label_id not in ID_TO_LABEL: continue
        label_name = ID_TO_LABEL[label_id]
        mask = np.not_equal(pred_masks[:, i], 0)
        if mask.sum() < opt["min_region_sizes"][0]: continue
        
        pred_inst = {
            "uuid": uuid4(),
            "confidence": float(pred["pred_scores"][i]),
            "vert_count": mask.sum(),
            "matched_gt": [],
        }
        for gti, gt in enumerate(gt2pred[label_name]):
            inter = np.logical_and(gt_ids == gt["instance_id"], mask).sum()
            if inter > 0:
                gt_copy = gt.copy(); gt_copy["intersection"] = inter
                pred_inst["matched_gt"].append(gt_copy)
                gt2pred[label_name][gti]["matched_pred"].append({
                    "uuid": pred_inst["uuid"],
                    "intersection": inter,
                    "confidence": pred_inst["confidence"],
                    "vert_count": mask.sum(),
                })
        pred2gt[label_name].append(pred_inst)
    return gt2pred, pred2gt

def evaluate(preds, gt_path, output_file, dataset="arabidopsis"):
    global CLASS_LABELS, VALID_CLASS_IDS, ID_TO_LABEL, LABEL_TO_ID, opt
    
    if dataset == "arabidopsis":
        opt["min_region_sizes"] = np.array([1])
        (
            CLASS_LABELS,
            VALID_CLASS_IDS,
            ID_TO_LABEL,
            LABEL_TO_ID,
        ) = _configure_arabidopsis_classes_from_label_db(gt_path)
        print(
            "[EvalScript] arabidopsis classes from label_db: "
            f"ids={VALID_CLASS_IDS.tolist()} labels={CLASS_LABELS}"
        )
    
    print(
        f"[EvalScript] dataset={dataset} preds={len(preds)} gt_path={gt_path} output_file={output_file}"
    )
    if not os.path.isdir(gt_path):
        raise OSError(f"GT directory does not exist: {gt_path}")

    matches = {}
    missing_gt_files = []
    used_scan_keys = []
    min_region_size = int(opt["min_region_sizes"][0])
    gt_instance_counter = defaultdict(int)
    pred_instance_counter = defaultdict(int)
    for scan_key, pred_item in preds.items():
        gt_file = os.path.join(gt_path, scan_key + ".txt")
        if os.path.isfile(gt_file):
            gt2pred, pred2gt = assign_instances_for_scan(
                pred_item, gt_file, scan_key=scan_key
            )
            matches[gt_file] = {"gt": gt2pred, "pred": pred2gt}
            used_scan_keys.append(scan_key)
            for label_name in CLASS_LABELS:
                gt_valid = [
                    gt for gt in gt2pred.get(label_name, [])
                    if int(gt.get("vert_count", 0)) >= min_region_size
                ]
                gt_instance_counter[label_name] += len(gt_valid)
                pred_instance_counter[label_name] += len(
                    pred2gt.get(label_name, [])
                )
        else:
            missing_gt_files.append(gt_file)

    if missing_gt_files:
        print(
            f"[EvalScript][WARN] missing_gt_files={len(missing_gt_files)} "
            f"sample={missing_gt_files[:5]}"
        )
    print(
        f"[EvalScript] used_scans={len(used_scan_keys)} "
        f"(from preds={len(preds)})"
    )
    if len(used_scan_keys) <= 100:
        print(f"[EvalScript] used_scan_keys={sorted(used_scan_keys)}")
    print(f"[EvalScript] total_gt_instances={dict(gt_instance_counter)}")
    print(f"[EvalScript] total_pred_instances={dict(pred_instance_counter)}")

    if len(matches) == 0:
        print("[EvalScript][WARN] No matched scan pairs found. Output metrics may be NaN.")

    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)
    
    # 按照 trainer.py 预期的格式写入 CSV 文件
    # 每一行: class, class_id, ap, ap50, ap25
    print_results(avgs)
    write_result_file(avgs, output_file)

    print(f"Evaluation complete. Result saved to {output_file}")
