# SDF-Net 数据重建运行顺序

## 0. 固定路径并检查空间

```bash
cd /ssd_data/lixiang_data/SDF-Net
conda activate pytorch

df -h /ssd_data /ssd_data2
du -sh /ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/*
```

原始数据只读目录：

```text
/ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID
```

重建数据输出目录：

```text
/ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data_Rebuild
```

不要将处理结果写回任一原始数据集目录。

## 1. 原始数据只读审计

先执行：

```bash
python audit_raw_cross_modal_datasets.py \
  --root /ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID \
  --output_dir ./logs/dataset_rebuild_audit \
  --check_images 200 \
  --write_image_manifest
```

输出文件：

```text
logs/dataset_rebuild_audit/audit.txt
logs/dataset_rebuild_audit/dataset_summary.csv
logs/dataset_rebuild_audit/image_manifest.csv
logs/dataset_rebuild_audit/candidate_pairs.csv
logs/dataset_rebuild_audit/bad_images.csv
logs/dataset_rebuild_audit/access_errors.csv
```

`candidate_pairs.csv` 只是路径和文件名启发式结果，不能直接作为训练清单。
运行后先分析各数据集的目录和真实配对规则，再生成第一阶段数据。

## 2. 审计已有第一阶段权重

```bash
python audit_pretrain_checkpoint.py \
  --config_file configs/SDF-Net_Multi_Paired.yml \
  --checkpoint /ssd_data/lixiang_data/SDF-Net/logs/Pretrain_TransOSS_Contrastive/transoss_contrastive_best.pth \
  --strict \
  --json ./logs/dataset_rebuild_audit/pretrain_checkpoint_audit.json
```

判断标准：

- `Missing SAR keys` 必须为空。
- RGB/SAR patch embedding 不应完全相同。
- 不应存在关键参数 shape mismatch。

旧的 `custom_vit_b_pretrain_ep100.pth` 已知缺少 SAR patch embedding，不作为最终第一阶段权重。

## 3. 审计原始 HOSS ReID 目录

如果审计结果确认 ReID 根目录是 `HOSS-ReID/HOSS`，执行：

```bash
python analyze_reid_dataset.py \
  --root /ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/HOSS-ReID/HOSS
```

若该路径不是直接包含 `bounding_box_train/query/bounding_box_test` 的目录，应以第 1 步报告中的实际路径替换。

## 4. MOS-Ship 无泄漏裁剪

先执行 dry-run：

```bash
python prepare_mos_ship_reid.py \
  --src_root /ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/MOS-Ship/MOS-Ship \
  --dst_root /ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data_Rebuild/MOS-Ship-ReID-SceneID-NoLeak \
  --identity_mode scene_object_id \
  --require_pair_for_train \
  --require_pair_for_eval \
  --drop_train_eval_overlap \
  --dry_run
```

检查 `Missing images=0`、`Malformed labels=0`，并确认训练和评估都有跨模态配对身份。
dry-run 通过后，去掉最后一行 `--dry_run`，使用完全相同的其他参数正式执行。

不要对已有非空输出目录直接重复运行。需要重建时先确认路径，再显式添加 `--overwrite`。

裁剪完成后审计：

```bash
python analyze_reid_dataset.py \
  --root /ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data_Rebuild/MOS-Ship-ReID-SceneID-NoLeak
```

必须满足：

```text
raw train & query   = 0
raw train & gallery = 0
raw query & gallery > 0
train pairs         > 0
```

## 5. 构建第一阶段配对数据

这一步必须在第 1 步报告分析完成后执行。针对每个数据集使用明确的官方配对键，生成：

```text
Cleaned_SDFNet_Data_Rebuild/Pretrain/opt/
Cleaned_SDFNet_Data_Rebuild/Pretrain/sar/
manifests/rebuild_pretrain_pairs.csv
```

禁止按排序位置配对，也禁止直接采用未经确认的 `candidate_pairs.csv`。原始数据审计完成后再确定该步骤的具体构建命令。

构建完成后执行：

```bash
python analyze_pretrain_dataset.py \
  --root /ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data_Rebuild/Pretrain \
  --pair_key relative \
  --duplicate_policy skip \
  --check_images \
  --max_check 500 \
  --write_manifest ./manifests/rebuild_pretrain_pairs.csv
```

## 6. 第一阶段训练或复用权重

如果第 2 步权重审计通过，可以先复用该权重进行第二阶段基线训练。

如果需要重新预训练，先做 smoke test：

```bash
python pretrain_transoss_contrastive.py \
  --config_file configs/pretrain_transoss_contrastive.yml \
  --data_root /ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data_Rebuild/Pretrain \
  --manifest ./manifests/rebuild_pretrain_pairs.csv \
  --output_dir ./logs/Pretrain_TransOSS_Contrastive_Rebuild_Smoke \
  --epochs 1 \
  --batch_size 16 \
  --max_pairs 512
```

smoke test 正常后正式训练：

```bash
nohup python pretrain_transoss_contrastive.py \
  --config_file configs/pretrain_transoss_contrastive.yml \
  --data_root /ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data_Rebuild/Pretrain \
  --manifest ./manifests/rebuild_pretrain_pairs.csv \
  --output_dir ./logs/Pretrain_TransOSS_Contrastive_Rebuild \
  --epochs 100 \
  > pretrain_transoss_contrastive_rebuild.log 2>&1 &
```

完成后再次运行第 2 步权重审计，将 checkpoint 改为：

```text
./logs/Pretrain_TransOSS_Contrastive_Rebuild/transoss_contrastive_best.pth
```

## 7. 合并 HOSS 与 MOS-Ship 第二阶段训练数据

先 dry-run：

```bash
python merge_reid_datasets.py \
  --source hoss=/ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/HOSS-ReID/HOSS \
  --source mos_ship=/ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data_Rebuild/MOS-Ship-ReID-SceneID-NoLeak \
  --dst_root /ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data_Rebuild/HOSS-MOSShip-Merged \
  --copy_mode copy \
  --eval_common_only \
  --dry_run
```

确认训练集与评估集 ID 无交集，并且 query/gallery ID 数量相同后，去掉 `--dry_run` 正式执行。

然后审计：

```bash
python analyze_reid_dataset.py \
  --root /ssd_data2/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data_Rebuild/HOSS-MOSShip-Merged
```

## 8. 第二阶段训练

建议依次训练三个对照模型：

1. HOSS only。
2. MOS-Ship only。
3. HOSS + MOS-Ship。

每个模型先将 `SOLVER.MAX_EPOCHS` 临时覆盖为 1 做 smoke test，再进行完整训练。训练配置中的 `DATASETS.ROOT_DIR` 和 `MODEL.PRETRAIN_PATH` 必须分别指向本次重建数据和通过审计的第一阶段权重。

## 9. 独立测试集重建与三模型测试

独立测试集的原始来源和转换规则要根据第 1 步报告确认。它不能参与第一阶段或第二阶段训练。

完成独立测试集重建和审计后，更新：

```text
configs/SDF-Net_Test_Independent.yml
configs/test_three_models_independent.json
```

最后执行：

```bash
python test_three_models_independent.py \
  --config configs/test_three_models_independent.json \
  --dry_run

python test_three_models_independent.py \
  --config configs/test_three_models_independent.json \
  --strict
```

最终主要比较 Rank-1、Rank-5 和 mAP；阈值 Top-1/Top-5、Coverage、Precision 和 Recall 作为补充指标。
