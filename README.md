# AtPOVcalculator
Point Cloud Processing Software for Arabidopsis Point Cloud Data

Preprocessing script:
plantdata\preprocess_data.py
Preprocesses the data by regularizing the original point cloud files according to data characteristics, mainly including:

Mapping NaN semantic labels to stem semantics and assigning semantic IDs and instance IDs.

Mapping non-NaN semantic labels to leaf semantics and assigning corresponding instance IDs.

Splitting the dataset into training and validation sets.

Generating relevant configuration files.

Preprocessing script:
plantdata\gt_gen.py
Preprocesses the data and generates ground truth label files.

Newly Added Model Adaptation File:
conf\data\arabidopsis.yaml
Main modifications: data path, data channels in_channels: 6,
num_labels: 3 # 0: Stem, 1: Leaf
num_workers: 4
batch_size: 4
test_batch_size: 1
The above parameters are adjusted according to hardware performance.
Important: voxel_size: 0.01 needs to be set.

Modifications to Mask3D Model Code:
benchmark\evaluate_semantic_instance.py
Modified sections:
L33-L88: Dynamically read categories from label_database.yaml instead of using fixed multi-classes from datasets such as ScanNet.
L310-L321: Added code for debug output information.

datasets\semseg.py
Data loader.
L31-L257: Specific adaptations for Arabidopsis, removing support for existing data formats like ScanNet and S3DIS.
This includes coordinate centering, color normalization, label organization, and feature concatenation.

trainer\trainer.py

Added visualization support, allowing only the export of result PLY files when VISUALIZATION_MINIMAL_MODE=true.
L62-L160: Added write_rgb_ply(), to_numpy_array(), and write_instance_semantic_ply() functions to export predictions and ground truth as PLY files.
L956-L1275: Enhanced save_visualizations() to export point cloud files including pred_instances.ply, pred_semantics.ply, and pred_point_labels.ply.

Post-processing support.
L396-L466: Read and validate post-processing related parameters.
L468-L510: Perform DBSCAN connected component segmentation on masks.
L533-L634: Reassign unassigned points to the nearest instance using KDTree.
L636-L954: Summarize the post-processing process, extract candidate instances from masks, perform connected component splitting, remove duplicates, filter out instances that are too small, backfill isolated points with instance labels, and generate unique pred_mask, pred_scores, and pred_classes.

Execution Workflow:
Data preprocessing: Use preprocess_data.py to generate NPY files that conform to the Mask3D format.
Use gt_gen.py to generate validation label files.

Train the model:
Use main_instance_segmentation.py to start training. The dataset, experiment name, and some configuration parameters need to be explicitly specified in the command.

bash
python main_instance_segmentation.py data=arabidopsis general.experiment_name="arabidopsis_train" data.data_dir="data/processed/arabidopsis" data.num_labels=2 general.num_targets=3
Inference: When save_visualizations=true is set, visualization point cloud files will be exported, generating instance segmentation, semantic segmentation results, and labeled segmentation results: saved/arabidopsis_eval/visualizations/<scene>/pred_point_labels.ply.

bash
python main_instance_segmentation.py data=arabidopsis general.train_mode=false general.experiment_name="arabidopsis_eval" general.checkpoint="saved/arabidopsis_train/last-epoch.ckpt" data.data_dir="data/processed/arabidopsis" data.num_labels=2 general.num_targets=3 general.save_visualizations=true
Feature extraction: Use the batch_extract_leaf_traits.py script to batch extract features. The scale parameter provides scale information; if coordinate scaling was applied during preprocessing, this can be used to revert it. This script generates a summary of the longest leaves in the root-dir directory and a feature table for each plant’s leaves in each <scene>.

bash
python batch_extract_leaf_traits.py `
  --root-dir "saved/arabidopsis_eval/visualizations" `
  --input-name "pred_point_labels.ply" `
  --scale 0.01 `
  --summary-name "longest_leaf_summary.csv"
