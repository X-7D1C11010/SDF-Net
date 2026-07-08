# 光学-SAR 跨模态 ReID 当前进展与技术方案

## 任务目标

本项目的目标是实现光学图像与 SAR 图像之间的跨模态目标匹配，也就是跨模态 ReID。当前任务有一个关键限制：训练集与独立测试集的类别 ID 不一致，因此测试阶段不能依赖模型输出的训练集分类结果来判断是否匹配。

最终测试流程应当以特征向量为核心：模型从光学图像和 SAR 图像中提取模态鲁棒的判别特征，然后根据特征相似度或距离完成排序、阈值匹配和 ReID 评估。

主要关注指标包括：

- Rank-1 / Top-1 准确率
- Rank-5 / Top-5 准确率
- mAP
- 正样本最近距离与负样本最近距离的差异
- positive margin rate

其中 Rank-1 / Top-1 是最直观的独立测试集匹配性能指标，Rank-5 和 mAP 用于辅助分析整体检索质量。

## 当前关键结论

### 1. 旧第一阶段预训练权重存在关键问题

使用 `audit_pretrain_checkpoint.py` 审计旧权重后得到如下结果：

```text
Checkpoint: /ssd_data/lixiang_data/SDF-Net/logs/Pretrain_Backbone_2/custom_vit_b_pretrain_ep100.pth
Raw keys: 150
Target backbone keys: 159
Missing SAR keys: ['patch_embed_SAR.proj.weight', 'patch_embed_SAR.proj.bias']
```

这说明旧权重没有真正包含 SAR 分支的 patch embedding 参数。此前第二阶段训练时，代码会把 RGB 分支的 `patch_embed` 复制到 `patch_embed_SAR` 作为初始化，但这不能等价于真正的光学-SAR 跨模态预训练。

因此，旧权重不能继续作为最终第一阶段预训练权重使用。后续必须重新进行包含 SAR 分支的 TransOSS/SDF-Net 第一阶段预训练。

### 2. 第二阶段 ReID 训练集规模过小

此前审计的第二阶段 ReID 训练集只有：

- 训练图像：266 张
- 训练 ID：154 个
- opt/SAR 成对 ID：112 个
- 部分 ID 只有单模态或单张图像

这个规模可以用于第二阶段适配，但不足以从零学习强泛化的光学-SAR ReID 特征空间。

因此，必须把非 ReID 的光学-SAR 配对数据用于第一阶段跨模态预训练，再用有限的 ReID 数据做第二阶段度量学习或弱分类头微调。

### 3. SEN1-2、DFC23、QXS-SAROPT、3MOS 等数据的定位

这类数据通常有光学/SAR 或多源遥感配对关系，但没有 ReID 所需的“同一目标跨模态身份 ID”标注。

因此它们适合用于：

- 第一阶段跨模态特征预训练
- 光学-SAR 模态对齐
- 训练 SAR 分支和光学分支共享/对齐的基础特征空间

它们不适合直接用于：

- 第二阶段 ReID ID 分类训练
- 训练集 ID 与测试集 ID 对应的类别匹配

除非某个数据集明确提供“同一目标跨模态身份 ID”，否则不能把它作为第二阶段 ReID 分类训练集。

### 4. 当前测试流程已经是特征相似度匹配

当前 `test_cross_modal.py` 的测试逻辑不是“先分类再匹配”，而是：

1. 加载训练好的模型权重。
2. 提取 query 和 gallery 的特征向量。
3. 计算 query-gallery 特征距离或相似度矩阵。
4. 根据特征排序、阈值匹配、Top-k 和 mAP 评估 ReID 性能。

分类头只在训练阶段作为塑造特征空间的监督信号使用。测试阶段不依赖训练类别 ID。

## 当前 manifest 分析

当前第一阶段预训练 manifest 文件为：

```text
manifests/all_pretrain_pairs.csv
```

统计结果如下：

- 配对总数：48,431 对
- 唯一 `pair_key`：48,431 个
- 唯一 opt 路径：48,431 个
- 唯一 sar 路径：48,431 个
- 空 opt 路径：0
- 空 sar 路径：0
- 重复 `pair_key`：0
- 重复 opt 路径：0
- 重复 sar 路径：0
- 图像格式分布：
  - `.png`：20,000 对
  - `.jpg`：28,431 对

从结构上看，这个 manifest 是干净的，可以直接作为 `pretrain_transoss_contrastive.py` 的输入。

不过需要注意：当前 manifest 中 `opt_source` 和 `sar_source` 全部为 `unknown`。原因是整合后的文件被平铺并重命名为如下形式：

```text
patch_000001_opt.png
patch_000001_sar.png
patch_000002_opt.png
patch_000002_sar.png
```

这种命名方式可以保证配对关系清晰，但会丢失原始数据集来源信息。因此，当前 manifest 无法证明 3MOS、HOSS-ReID OptiSar_Pair、MOS-Ship、Multi-Resolution-SAR-dataset、OSdataset、OSdataset2.0、OsEval、QXS-SAROPT 是否全部被纳入。

如果后续需要检查每个来源是否遗漏，建议保留来源目录结构，例如：

```text
Pretrain/
  opt/3MOS/...
  opt/QXS-SAROPT/...
  opt/MOS-Ship/...
  sar/3MOS/...
  sar/QXS-SAROPT/...
  sar/MOS-Ship/...
```

然后重新运行：

```bash
python analyze_pretrain_dataset.py \
  --root /ssd_data/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data/Pretrain \
  --expected_sources 3MOS OptiSar_Pair MOS-Ship Multi-Resolution-SAR-dataset OSdataset OSdataset2.0 OsEval QXS-SAROPT \
  --pair_key relative \
  --write_manifest manifests/all_pretrain_pairs.csv \
  --check_images \
  --max_check 300
```

如果当前平铺命名是最终版本，也可以继续使用，但建议在数据整合阶段额外保存一个来源索引文件，例如：

```text
pair_key, opt_path, sar_path, source_dataset
```

否则后续无法从文件名反推出原始数据来源。

## 当前方法路线

### 第一阶段：跨模态特征预训练

第一阶段使用没有 ReID 身份标签、但具有光学/SAR 配对关系的数据进行预训练。目标是让 TransOSS/SDF-Net backbone 学到基础的光学-SAR 对齐特征，并真正训练 `patch_embed_SAR`。

推荐运行命令：

```bash
python pretrain_transoss_contrastive.py \
  --config_file configs/pretrain_transoss_contrastive.yml \
  --manifest manifests/all_pretrain_pairs.csv \
  --init_weight /ssd_data/lixiang_data/SDF-Net/model/vit_base_p16_224.pth
```

该脚本直接训练 `vit_base_patch16_224_TransOSS` backbone，而不是普通 timm ViT。因此保存出的权重应包含：

- `patch_embed.proj.weight`
- `patch_embed.proj.bias`
- `patch_embed_SAR.proj.weight`
- `patch_embed_SAR.proj.bias`
- Transformer blocks
- MIE/SSE/解耦相关模块参数

预期输出：

```text
logs/Pretrain_TransOSS_Contrastive/transoss_contrastive_best.pth
logs/Pretrain_TransOSS_Contrastive/transoss_contrastive_last.pth
```

第一阶段训练完成后必须审计新权重：

```bash
python audit_pretrain_checkpoint.py \
  --config_file configs/SDF-Net_Multi_Paired.yml \
  --checkpoint ./logs/Pretrain_TransOSS_Contrastive/transoss_contrastive_best.pth \
  --strict
```

如果新权重仍然缺少 `patch_embed_SAR.proj.weight` 或 `patch_embed_SAR.proj.bias`，说明第一阶段预训练链路仍然没有修正。

### 第二阶段：ReID 特征空间适配

第二阶段只能使用具有真实跨模态 ReID 身份 ID 的数据，例如 HOSS 或当前整理出的 ReID 成对数据。

当前提供两种配置：

#### 方案 A：弱分类头训练

```bash
python train.py --config_file configs/SDF-Net_Multi_Paired.yml
```

该配置保留较小的 ID 分类损失，同时加入 triplet、跨模态 contrastive、prototype 和结构一致性约束。

适用情况：

- 少量 ID 分类监督仍然能帮助特征聚类。
- 训练集与测试集虽然类别不同，但训练 ID 仍可作为特征空间监督。

#### 方案 B：纯度量学习训练

```bash
python train.py --config_file configs/SDF-Net_Multi_MetricOnly.yml
```

该配置设置：

```yaml
ID_LOSS_WEIGHT: 0.0
```

也就是不使用 ID 分类损失，只保留特征空间相关约束。

适用情况：

- 分类头在小训练集上过拟合明显。
- 希望尽量减少训练类别对特征空间的绑定。
- 更强调跨模态相似性匹配，而不是训练 ID 分类。

当前数据加载与采样已经支持：

```yaml
DATASETS.TRAIN_PAIR_ONLY: True
DATALOADER.CROSS_MODAL_SAMPLER: True
```

作用是：

- 过滤掉没有 opt/SAR 成对样本的训练 ID。
- batch 内尽量保证每个采样 ID 同时包含 opt 和 sar。
- 提高 `CMCon`、`CMProto`、`Struct` 等跨模态损失的有效性。

训练日志现在会输出：

```text
ID
Tri
CMCon
CMProto
Orth
Struct
```

如果 `CMCon` 或 `CMProto` 长期接近 0，说明跨模态监督没有真正起作用，需要检查 sampler、batch 组成或训练数据配对。

### 第三阶段：不依赖类别的特征匹配

测试阶段不使用分类头输出，而是直接基于特征相似度进行匹配。

推荐先比较不同距离/相似度：

```bash
python test_cross_modal.py \
  --config_file configs/SDF-Net_Test.yml \
  --weight_path /path/to/best.pth \
  --compare_metrics \
  --comparison_metrics cosine_distance csls_similarity hybrid \
  --csls_k 10 \
  --topk 20
```

当前支持的主要匹配方式：

- `cosine_distance`
- `cosine_similarity`
- `csls_similarity`
- `csls_distance`
- `hybrid`
- 自适应阈值匹配
- mutual top-k 双向一致性匹配

如果要使用更保守的双向一致性匹配：

```bash
python test_cross_modal.py \
  --config_file configs/SDF-Net_Test.yml \
  --weight_path /path/to/best.pth \
  --distance_metric csls_similarity \
  --csls_k 10 \
  --require_mutual \
  --mutual_k 5 \
  --topk 20
```

其中 CSLS 用于缓解跨模态检索中的 hubness 问题，也就是某些 gallery 样本容易成为大量 query 的错误近邻。mutual top-k 用于保留双向都认为相近的候选匹配。

## 结果分析方法

独立测试集上应重点观察：

- Rank-1
- Rank-5
- mAP
- Threshold Top-1
- Threshold Top-5
- Best-pos mean
- Best-neg mean
- Margin mean
- Positive margin rate
- Separation score

如果出现：

```text
Best-pos mean > Best-neg mean
Margin mean < 0
Positive margin rate 很低
```

说明同 ID 的光学/SAR 样本在特征空间中没有靠近，模型特征本身仍然没有完成跨模态对齐。这种情况下，继续调阈值或换匹配算法只能带来有限提升，主要瓶颈仍然是第一阶段预训练和第二阶段特征适配。

如果 CSLS 相比 cosine 有提升，说明局部密度校正有效，跨模态 hubness 是问题之一。

如果 mutual top-k 提升 precision 但 coverage 很低，说明模型只在少数样本上有较稳定的双向匹配，大多数样本仍然特征不可靠。

## 当前风险点

### 1. 第一阶段数据来源不可追踪

当前 manifest 有 48,431 对干净配对，但来源全部为 `unknown`。这意味着不能从当前 manifest 判断各原始数据集是否遗漏。

建议后续保留来源目录或来源索引。

### 2. 第一阶段数据不等价于 ReID 数据

光学/SAR 场景配对数据可以帮助学习模态对齐，但不能替代第二阶段 ReID 身份监督。

第一阶段学习的是“跨模态通用特征”，第二阶段才学习“ReID 身份判别特征”。

### 3. 第二阶段 ReID 数据仍然是主要瓶颈

当前真正可用于第二阶段训练的跨模态 ReID 数据仍然很少。若希望独立测试集 Rank-1 达到较高水平，仍需要更多接近目标域的跨模态 ReID 身份标注数据。

在没有更多 ReID 数据的情况下，只能依靠：

- 更强的第一阶段跨模态预训练
- paired-only 采样
- metric-only 或弱分类头训练
- CSLS 特征匹配
- mutual top-k 双向一致性
- 可选的无监督目标域伪配对适配

### 4. 旧 `pretrain_log.txt` 不是新预训练流程的日志

当前看到的 `pretrain_log.txt` 属于旧流程，日志中包含：

```text
configs/pretrain_transoss.yml
patch_embed_SAR not in the pth
Number of RGB-SAR pair: 55654
```

这不是新脚本 `pretrain_transoss_contrastive.py` 的日志。

新流程日志应包含类似：

```text
transoss.pretrain
Pretrain pairs: 48431
infonce=...
struct=...
transoss_contrastive_best.pth
```

因此后续应以新脚本产生的日志和 checkpoint 为准。

## 推荐后续步骤

1. 如有可能，恢复或保留第一阶段数据来源信息。
2. 使用当前 `manifests/all_pretrain_pairs.csv` 运行 `pretrain_transoss_contrastive.py`。
3. 审计 `transoss_contrastive_best.pth`，确认包含 SAR 分支参数。
4. 分别训练：

```bash
python train.py --config_file configs/SDF-Net_Multi_Paired.yml
python train.py --config_file configs/SDF-Net_Multi_MetricOnly.yml
```

5. 分别在独立测试集上比较：

```bash
python test_cross_modal.py \
  --config_file configs/SDF-Net_Test.yml \
  --weight_path /path/to/best.pth \
  --compare_metrics \
  --comparison_metrics cosine_distance csls_similarity hybrid \
  --csls_k 10 \
  --topk 20
```

6. 对最佳模型运行 mutual top-k 匹配：

```bash
python test_cross_modal.py \
  --config_file configs/SDF-Net_Test.yml \
  --weight_path /path/to/best.pth \
  --distance_metric csls_similarity \
  --csls_k 10 \
  --require_mutual \
  --mutual_k 5 \
  --topk 20
```

7. 根据 distance diagnostics 判断主要瓶颈：

- 如果同 ID 样本仍然比不同 ID 样本更远，优先改进预训练和特征学习。
- 如果排序已经改善但阈值指标差，优先优化匹配阈值、CSLS 参数和 mutual top-k。
