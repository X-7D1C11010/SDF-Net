# SDF-Net ReID 优化与评估排查报告

本文档面向当前仓库实现，说明代码结构、数据流、指标计算逻辑、两次测试结果异常的原因，以及本次已经完成的优化修改和后续测试指南。

## 1. 代码结构与数据流

### 1.1 训练流程

入口文件是 `train.py`。

数据流如下：

1. 读取 `configs/*.yml`，合并命令行覆盖项。
2. `datasets.make_dataloader.make_dataloader()` 构造 `MergedDataset`、训练增强、验证集 dataloader。
3. `model.make_model.make_model()` 构建 SDF-Net Transformer。
4. `loss.make_loss.make_loss()` 组合 ID loss、triplet loss、structure loss。
5. `processor.processor.do_train()` 执行混合精度训练，并按 `EVAL_PERIOD` 在 query/gallery 上计算 mAP 和 CMC。

核心训练损失位于 `processor/processor.py`：

- `score_fuse` 用于 ID 分类损失。
- `feat_fuse` 用于 triplet metric loss。
- `feat_shared` 和 `feat_spec` 用正交约束降低冗余。
- `f_struct` 用结构一致性损失增强跨模态稳定性。

本次排查发现历史日志中曾出现 `train | 0 ids | 0 images`，这会导致训练 epoch 看似正常结束，但模型实际上没有学习。现在 dataloader 会在训练集为空、query/gallery 为空、query/gallery 无重叠 ID 时直接报错。

### 1.2 模型与特征提取

模型主体在 `model/make_model.py`，Transformer backbone 来自 `model/backbones/vit_transoss.py`。当前配置使用：

- ViT base patch16。
- MIE 模态/相机嵌入。
- SSE 结构增强。
- DISENTANGLE 分支，将 shared feature 和 modality-specific feature 融合。
- 推理阶段返回 `bottleneck_fuse(feat_fuse)`，也就是分类头之前的 ReID embedding。

跨数据集标签不匹配场景下，分类头类别数不可复用，因此 `utils/feature_extractor.py` 用 `num_classes=1` 初始化模型，加载 checkpoint 时跳过 classifier 和 `logit_scale`，只保留 backbone、BN neck 和特征提取相关参数。

本次增强了 checkpoint 加载诊断：现在会打印加载成功、部分加载、shape mismatch、unexpected key 和 missing non-head key 的数量，便于定位模型结构与权重不匹配。

### 1.3 原始测试流程

`test.py` 使用 `processor.do_inference()`：

1. 按 query + gallery 顺序提取特征。
2. `utils.metrics.R1_mAP_eval.compute()` 归一化特征。
3. 默认用欧氏距离计算 distmat。
4. `eval_func()` 基于标签计算 CMC 和 mAP。

这条路径适合常规 ReID 排序评估，但不提供阈值匹配结果，也不区分跨数据集标签不匹配时的“特征匹配”与“标签分类”问题。

### 1.4 跨模态测试流程

`test_cross_modal.py` 是标签无关跨模态匹配入口：

1. 加载 dataloader 和模型。
2. `FeatureExtractor.extract()` 提取 L2 normalized feature。
3. `CrossModalMatcher.fit()` 基于 query 到 gallery 的每个 query 最近距离做无监督阈值校准。
4. `CrossModalMatchingPipeline.evaluate()` 计算阈值匹配、Top-1/Top-5 阈值准确率、mAP、CMC Rank-1/5/10/20。
5. 保存 `metrics.json` 和 `top_matches.csv`。

默认距离度量是 `cosine_distance = 1 - cosine_similarity`。因为特征已经 L2 归一化，该度量对特征尺度变化更鲁棒，是当前推荐默认值。

## 2. 两次结果异常分析

你给出的两次结果分别是：

- 非 mutual：Accuracy 0.9233，Precision 0.0327，Recall 0.0320，mAP 0.0556，Rank-1 0.0267，Rank-5 0.0499。
- mutual top-1：Accuracy 0.9599，Precision 0.0753，Recall 接近 0，mAP 0.0576，Rank-1 0.0259，Rank-5 0.0494。

### 2.1 Accuracy 高不是模型好

当前测试规模约为：

- Query: 26717
- Gallery: 9214
- 总 pair 数: 26717 * 9214 = 246170438

从 precision/recall 反推，正样本 pair 约 986 万，占比约 4%。也就是说，全部判为“不匹配”也能得到约 96% accuracy。因此 92% 到 96% 的二分类 accuracy 主要反映类别极不平衡，不应作为 ReID 主指标。

本次已在输出中增加：

- `Positive rate`
- `All-neg baseline`
- `Balanced Acc`

用来直接提示 accuracy 是否被负样本主导。

### 2.2 ReID 指标低说明排序能力弱

mAP 和 Rank-k 只关心每个 query 的 gallery 排序质量，不能通过大量 true negative 抬高。你当前 Rank-1 约 2.6%，若测试集中有效身份约 44 个，随机 Rank-1 基线约为 1/44 = 2.27%。因此当前模型在这个大测试集上接近随机检索。

这通常不是阈值问题，而是特征空间没有把跨模态同 ID 拉近。优先检查：

- 训练集是否为空或路径不一致。
- 测试 checkpoint 是否真的是有效训练权重。
- checkpoint 与当前模型配置是否匹配。
- query/gallery PID 是否被正确解析。
- 测试集是否与训练数据分布差异过大。

### 2.3 `--require_mutual` 为什么几乎不改变 mAP/Rank

`--require_mutual` 只影响阈值接受矩阵：一个 pair 既要过阈值，又要互为 top-1 才被接受。

mAP/Rank-1/Rank-5 是直接从距离矩阵排序得到的，不依赖 `match_matrix`。因此两次运行 mAP/Rank 接近是合理的。mutual 只会大幅减少 accepted pairs，提高一点 precision，但 recall 会极低。

### 2.4 两次阈值不同的可疑原因

只切换 `--require_mutual` 理论上不应改变阈值校准，因为阈值来自距离矩阵本身。阈值差异说明两次运行的特征或 PID 可能不完全一致。主要风险点：

- 模型未设随机种子时，未加载到的层保持随机初始化。
- checkpoint 部分加载失败但日志不明显。
- `MergedDataset._extract_pid()` 原来使用 Python 内置 `hash()` fallback，跨进程不稳定。
- 文件名不匹配正则时，同一身份的 opt/sar 可能被解析成不同 PID。

本次已经增加随机种子设置、checkpoint 加载诊断，并把 PID fallback 改为稳定解析。

## 3. 本次代码修改

### 3.1 `utils/cross_modal_matching.py`

新增和优化内容：

- 增加 `threshold_topk` 指标，包含 Top-1 和 Top-5 阈值准确率。
- 增加 `Top-k Coverage` 和 `Top-k Precision`，用于区分“没过阈值”和“排序错误”。
- 增加 `positive_rate`、`negative_rate`、`all_negative_accuracy`、`balanced_accuracy`。
- 阈值校准后的距离矩阵在评估阶段复用，避免重复计算。
- mAP/CMC、threshold Top-k、top_matches 共用一次 `argsort` 结果，避免三次排序。
- 默认不再保存完整 `.npy` 距离矩阵，避免大规模测试时写出数 GB 文件。

阈值 Top-k 定义：

- `Top-1 Acc`: rank-1 gallery 是同 ID 且被阈值接受的 query 比例。
- `Top-5 Acc`: top-5 中存在同 ID 且被阈值接受的 query 比例。
- `Top-k Coverage`: top-k 中至少有一个 gallery 被阈值接受的 query 比例。
- `Top-k Precision`: top-k 内被阈值接受 pair 中真实正样本的比例。

ReID 排序指标仍然使用传统 mAP 和 CMC Rank-k。

### 3.2 `test_cross_modal.py`

新增参数：

- `--seed`: 在模型初始化和特征提取前设置随机种子。
- `--save_matrices`: 需要调试完整矩阵时才保存 `.npy`。

输出摘要现在包含：

- mAP
- Rank-1
- Threshold Top-1
- Threshold Top-5
- Precision
- F1

### 3.3 `datasets/make_dataloader.py`

新增数据合法性检查：

- query 为空时报错。
- gallery 为空时报错。
- query/gallery 无重叠 ID 时报错。
- 训练集为空或训练 ID 为 0 时报错。
- pair training 没有 opt/sar 配对时报错。

这可以尽早暴露路径或文件命名问题，避免训练和测试结果失真。

### 3.4 `datasets/merged.py`

新增 query 和 gallery 目录存在性检查。

PID 解析改为：

1. 优先解析 `0001_s1c1_opt`、`0001_s1c1_sar` 这类标准文件名。
2. 支持大小写不敏感。
3. 支持 `0001_xxx`、`0001`、`pid0001`、`id0001`。
4. 最后才对去除 modality 后的 stem 做稳定 MD5 fallback。

这修复了 Python `hash()` 跨进程随机化和 opt/sar 被解析成不同 ID 的风险。

### 3.5 `model/make_model.py`

增强 `build_transformer.load_param()`：

- 跳过分类头和 `logit_scale` 时计数。
- 记录完全加载、部分加载、shape mismatch、unexpected key。
- 打印 missing non-head keys。

如果看到大量 non-head missing key，应优先怀疑 checkpoint 与当前配置不匹配。

## 4. 指标计算逻辑

### 4.1 Binary Pair Metrics

构造所有 query/gallery pair：

- `gt[q, g] = 1` 当 `q_pid == g_pid`。
- `pred[q, g] = 1` 当距离过阈值。

计算：

- Accuracy = `(TP + TN) / all pairs`
- Precision = `TP / (TP + FP)`
- Recall = `TP / (TP + FN)`
- F1 = `2PR / (P + R)`
- Balanced Acc = `(Recall + Specificity) / 2`

注意：pair 数极大且负样本占绝大多数时，Accuracy 不是主指标。

### 4.2 Threshold Top-1/Top-5

先按距离从小到大排序，再只看 top-k：

- top-k 中存在同 ID 且被阈值接受，则该 query 命中。
- 该指标同时约束排序质量和阈值接受策略。

它比全 pair accuracy 更接近实际“给定 query 找人”的任务。

### 4.3 mAP 和 CMC

对每个 query 排序所有 gallery：

- CMC Rank-k: 正确 ID 是否出现在前 k。
- AP: 每个正样本位置处 precision 的平均。
- mAP: 所有有效 query 的 AP 平均。

这是 ReID 领域最重要的主指标，建议用 mAP、Rank-1、Rank-5 作为模型性能判断依据。

## 5. 推荐测试方案

### 5.1 数据目录

`MergedDataset` 期望目录结构：

```text
Merged/
  bounding_box_train/
    opt/
      0001_s1c1_opt.jpg
    sar/
      0001_s2c1_sar.jpg
  query/
    0001_s1c1_opt.jpg
  bounding_box_test/
    0001_s2c1_sar.jpg
```

建议文件名优先使用 `数字ID_s数字c数字_opt/sar`。如果使用自定义命名，请确认 query/gallery 同一身份能解析为同一个 PID。

### 5.2 环境依赖

仓库 `requirements.txt` 已包含主要依赖：

```bash
pip install -r requirements.txt
```

如需使用 `--classifier_type svm/logistic_regression/random_forest` 这类有监督消融分类器，还需要：

```bash
pip install scikit-learn
```

默认无监督阈值匹配不需要 scikit-learn。

### 5.3 训练

先确认 `configs/SDF-Net_Multi.yml` 中：

- `DATASETS.ROOT_DIR` 指向真实训练数据。
- `MODEL.PRETRAIN_PATH` 指向存在的预训练 backbone。
- `OUTPUT_DIR` 可写。

运行：

```bash
python train.py --config_file configs/SDF-Net_Multi.yml
```

训练日志必须看到：

- train ids 大于 0。
- train images 大于 0。
- loss 非 0。
- validation mAP 和 Rank-1 正常更新。
- `best.pth` 被保存。

### 5.4 常规 ReID 测试

```bash
python test.py --config_file configs/SDF-Net_Test.yml TEST.WEIGHT /path/to/best.pth
```

该流程输出 mAP、Rank-1、Rank-5、Rank-10，是对纯排序能力的基线检查。

### 5.5 跨模态阈值匹配测试

推荐默认命令：

```bash
python test_cross_modal.py --config_file configs/SDF-Net_Test.yml --seed 1949 TEST.WEIGHT /path/to/best.pth
```

如果只想比较不同距离：

```bash
python test_cross_modal.py --config_file configs/SDF-Net_Test.yml --seed 1949 --compare_metrics TEST.WEIGHT /path/to/best.pth
```

如果要保存完整矩阵用于离线分析：

```bash
python test_cross_modal.py --config_file configs/SDF-Net_Test.yml --seed 1949 --save_matrices TEST.WEIGHT /path/to/best.pth
```

注意：26717x9214 的 float32 矩阵约 984 MB，保存 metric、distance、match 三个矩阵会占用数 GB。

### 5.6 阈值策略建议

默认：

```bash
--distance_metric cosine_distance --threshold_strategy mad --threshold_mad_scale 3.0
```

调参方向：

- precision 过低、accepted pairs 过多：降低 `threshold_mad_scale` 或改用较低 percentile。
- recall 过低、accepted pairs 过少：提高 `threshold_mad_scale` 或 percentile。
- 只做高置信一对一候选：使用 `--require_mutual`，但不要把它作为召回主评估。

可选命令：

```bash
python test_cross_modal.py --config_file configs/SDF-Net_Test.yml --threshold_strategy percentile --threshold_percentile 50
```

```bash
python test_cross_modal.py --config_file configs/SDF-Net_Test.yml --threshold_mad_scale 1.5
```

## 6. 预期效果与判断标准

完成有效训练和修复数据解析后，预期现象：

- Rank-1 应显著高于随机基线。例如 44 个有效 ID 时，随机 Rank-1 约 2.27%。
- mAP 应随 Rank-1 同步提升，不应长期停留在 5% 左右。
- Threshold Top-1/Top-5 应与 Rank-1/Rank-5 同向变化。
- Binary Pair Accuracy 可能下降，但 Precision、Recall、F1、Balanced Acc 应更有解释力。
- `--require_mutual` 的 accepted pairs 会明显减少，precision 可能上升，recall 会下降。

建议主报告使用：

- mAP
- Rank-1
- Rank-5
- Threshold Top-1 Acc
- Threshold Top-5 Acc
- Threshold Top-k Coverage
- Precision/Recall/F1
- Positive rate 和 all-negative baseline

## 7. 常见问题与解决方案

### 7.1 训练集为空

现象：启动后直接报 `Training set is empty`。

处理：

- 检查 `DATASETS.ROOT_DIR`。
- 检查是否存在 `bounding_box_train/opt` 和 `bounding_box_train/sar`。
- 检查文件扩展名是否为 jpg、jpeg、png。

### 7.2 Query/Gallery 无重叠 ID

现象：报 `No overlapping IDs between query and gallery`。

处理：

- 检查 query 和 gallery 文件名是否包含一致身份 ID。
- 使用标准命名 `0001_s1c1_opt.jpg` 和 `0001_s2c1_sar.jpg`。

### 7.3 mAP/Rank 接近随机

处理顺序：

1. 检查训练日志是否真的有训练样本和非零 loss。
2. 检查 `TEST.WEIGHT` 是否是对应数据集的 `best.pth`。
3. 查看加载日志是否有大量 missing non-head keys。
4. 检查 query/gallery PID 解析是否正确。
5. 使用 `--compare_metrics` 确认是否只是距离度量不适配。

### 7.4 Accuracy 高但 Precision/F1 极低

这是全 pair 二分类的类别不平衡问题。看 `Positive rate` 和 `All-neg baseline`，并以 mAP、Rank-1、Rank-5、Threshold Top-1/Top-5 为主。

### 7.5 `--require_mutual` 后 Recall 接近 0

mutual top-1 是高精度低召回策略。它适合输出少量高置信候选，不适合衡量整体 ReID 检索能力。

### 7.6 跑得慢或磁盘占用大

处理：

- 默认不要加 `--save_matrices`。
- 优先使用 `cosine_distance` 或 `euclidean`。
- 避免在大规模数据上使用 `manhattan`、`chebyshev`、`minkowski`、`mahalanobis`，这些实现会产生更高内存压力。

后续可进一步优化：

- 分块计算距离矩阵。
- 使用 FAISS 做近邻检索。
- top-k 场景用 `argpartition` 替代完整排序。
- 对 mAP/CMC 保留完整排序或采用可验证的近似评估。
