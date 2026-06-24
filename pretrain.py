# -*- coding: utf-8 -*-
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from PIL import Image
import torchvision.transforms as T
import timm

# ================= 配置日志 =================
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= 数据集定义 =================
class CrossModalPretrainDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.opt_dir = os.path.join(data_root, 'opt')
        self.sar_dir = os.path.join(data_root, 'sar')
        self.transform = transform
        
        # 获取所有文件
        opt_files = os.listdir(self.opt_dir)
        sar_files = os.listdir(self.sar_dir)
        
        # 通过去掉 '_opt' 和 '_sar' 标识，提取出核心ID (如 'patch_000001')
        opt_dict = {f.split('_opt')[0]: f for f in opt_files if '_opt' in f}
        sar_dict = {f.split('_sar')[0]: f for f in sar_files if '_sar' in f}
        
        # 取交集寻找匹配的核心ID
        common_keys = sorted(list(set(opt_dict.keys()) & set(sar_dict.keys())))
        self.paired_files = [(opt_dict[k], sar_dict[k]) for k in common_keys]
        
        logger.info(f"检测到严格空间对齐的预训练图像对数量: {len(self.paired_files)}")
        
    def __len__(self):
        return len(self.paired_files)
        
    def __getitem__(self, idx):
        # 根据配对好的文件名分别读取
        opt_filename, sar_filename = self.paired_files[idx]
        opt_path = os.path.join(self.opt_dir, opt_filename)
        sar_path = os.path.join(self.sar_dir, sar_filename)
        
        opt_img = Image.open(opt_path).convert('RGB')
        sar_img = Image.open(sar_path).convert('RGB')
        
        if self.transform:
            opt_img = self.transform(opt_img)
            sar_img = self.transform(sar_img)
            
        return opt_img, sar_img

# ================= 对比学习损失 (InfoNCE) =================
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features_opt, features_sar):
        # L2 归一化
        features_opt = F.normalize(features_opt, dim=-1)
        features_sar = F.normalize(features_sar, dim=-1)
        
        # 计算相似度矩阵 (Batch_size x Batch_size)
        logits = torch.matmul(features_opt, features_sar.T) / self.temperature
        
        # 标签为对角线 (即 [0, 1, 2, ..., N-1])
        batch_size = features_opt.shape[0]
        labels = torch.arange(batch_size, device=features_opt.device)
        
        # 双向交叉熵 (Opt->SAR 和 SAR->Opt)
        loss_o2s = F.cross_entropy(logits, labels)
        loss_s2o = F.cross_entropy(logits.T, labels)
        
        return (loss_o2s + loss_s2o) / 2.0

# ================= 预训练主循环 =================
def do_pretrain():
    # --- 1. 参数配置 ---
    DATA_ROOT = "/ssd_data/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data/Pretrain"
    OUTPUT_DIR = "./logs/Pretrain_Backbone_2"
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 3e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 2. 数据加载 ---
    transform = T.Compose([
        T.Resize((224, 224)), # ViT 标准输入尺寸
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CrossModalPretrainDataset(DATA_ROOT, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    
    # --- 3. 模型构建 (使用 ViT-Base 作为底座) ---
    logger.info("构建无头 Vision Transformer 底座...")
    
    # 将 pretrained 改为 False，告诉代码不要去网上下载
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    model.to(DEVICE)
    
    # 加载你预先下载好的本地权重
    pretrain_path = "/ssd_data/lixiang_data/SDF-Net/model/vit_b512_pre.pth"
    if os.path.exists(pretrain_path):
        logger.info(f"正在加载本地预训练权重: {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location=DEVICE)
        
        # 为了防止原权重里带有分类头（如 head.weight）导致维度报错，进行安全过滤
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        # 更新并加载权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"成功导入 {len(pretrained_dict)} 个匹配的权重层！")
    else:
        logger.warning(f"未找到本地权重文件 {pretrain_path}，将使用随机初始化开始训练！")
    
    # --- 4. 优化器与损失函数 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    criterion = ContrastiveLoss(temperature=0.07)
    scaler = amp.GradScaler()
    
    # --- 5. 训练循环 ---
    logger.info(">>> 开始跨模态对比学习预训练 <<<")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0
        
        for step, (opt_img, sar_img) in enumerate(dataloader):
            opt_img, sar_img = opt_img.to(DEVICE), sar_img.to(DEVICE)
            optimizer.zero_grad()
            
            with amp.autocast(enabled=True):
                # 共享权重提取特征
                feat_opt = model(opt_img)
                feat_sar = model(sar_img)
                loss = criterion(feat_opt, feat_sar)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            if step % 20 == 0:
                logger.info(f"Epoch [{epoch}/{EPOCHS}] Step [{step}/{len(dataloader)}] Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - start_time
        logger.info(f"==> Epoch [{epoch}] 结束 | 平均 Loss: {avg_loss:.4f} | 耗时: {epoch_time:.1f}s")
        
        # 每 10 轮保存一次底座权重
        if epoch % 10 == 0:
            save_path = os.path.join(OUTPUT_DIR, f"custom_vit_b_pretrain_ep{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"权重已保存至: {save_path}")

    logger.info("预训练完成！")

if __name__ == "__main__":
    do_pretrain()