# 拟南芥分割及参数提取

这个项目用于拟南芥点云实例分割，训练/推理入口是 `main_instance_segmentation.py`，叶片特征提取入口是：

- `extract_leaf_traits_from_ply.py`
- `batch_extract_leaf_traits.py`

## 环境

```powershell
conda env create -f environment.yml
conda activate mask3d_cuda113
```

如果缺少 CUDA 扩展，可额外编译：

```powershell
cd third_party/pointnet2
python setup.py install
cd ../../utils/pointops2
python setup.py install
cd ../..
```

## 训练

默认数据目录是 `data/processed/arabidopsis`。

示例：

```powershell
python main_instance_segmentation.py `
  data=arabidopsis `
  general.experiment_name="arabidopsis_train" `
  data.data_dir="data/processed/arabidopsis" `
  data.num_labels=2 `
  general.num_targets=3
```

训练结果默认保存在：

```text
saved/arabidopsis_train
```

## 推理

示例：

```powershell
python main_instance_segmentation.py `
  data=arabidopsis `
  general.train_mode=false `
  general.experiment_name="arabidopsis_eval" `
  general.checkpoint="saved/arabidopsis_train/last-epoch.ckpt" `
  data.data_dir="data/processed/arabidopsis" `
  data.num_labels=2 `
  general.num_targets=3 `
  general.save_visualizations=true
```

如果打开 `general.save_visualizations=true`，会生成：

```text
saved/arabidopsis_eval/visualizations/<scene>/pred_point_labels.ply
```

后面的特征提取脚本就是读取这个文件。

## 提取叶片特征

单株示例：

```powershell
python extract_leaf_traits_from_ply.py `
  --input "saved/arabidopsis_eval/visualizations/plant_001/pred_point_labels.ply" `
  --output-dir "saved/arabidopsis_eval/visualizations/plant_001/traits" `
  --scale 1.0
```

批量示例：

```powershell
python batch_extract_leaf_traits.py `
  --root-dir "saved/arabidopsis_eval/visualizations" `
  --input-name "pred_point_labels.ply" `
  --scale 1.0 `
  --summary-name "longest_leaf_summary.csv"
```