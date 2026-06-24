# -*- coding: utf-8 -*-
import os
import re
import cv2
import random
import logging
from pathlib import Path
from collections import defaultdict

# ================= 配置日志 =================
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Dynamic_ReID_Appender:
    def __init__(self, raw_mos_path, existing_reid_path, rollback_pid=None, train_ratio=0.5, query_modality='sar', padding=10):
        self.raw_data_root = Path(raw_mos_path)
        self.existing_reid_dir = Path(existing_reid_path)
        self.train_ratio = train_ratio
        self.query_modality = query_modality.lower()
        self.padding = padding
        
        # 绑定重识别数据集标准目录结构
        self.dirs = {
            'train': self.existing_reid_dir / 'bounding_box_train',
            'query': self.existing_reid_dir / 'query',
            'gallery': self.existing_reid_dir / 'bounding_box_test'
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        self.all_crops = defaultdict(list)
        
        # 1. 执行自动回滚机制：清理上次产生标签冲突和模态丢失的脏数据
        if rollback_pid is not None:
            self.cleanup_previous_errors(rollback_pid)
            
        # 2. 重新扫描纯净的图库，计算安全的全局 ID 新起点
        self.start_pid = self._get_max_existing_pid() + 1

    def cleanup_previous_errors(self, start_pid):
        """物理删除上次错误整合的全部垃圾文件，使数据集无损回滚"""
        logger.info(f"=== 🧹 触发数据回滚：正在清除全局 ID >= {start_pid} 的冲突图片 ===")
        count = 0
        for sub_dir in self.dirs.values():
            if not sub_dir.exists(): continue
            for img_path in sub_dir.glob("*.*"):
                match = re.match(r'^(\d+)_', img_path.name)
                if match:
                    pid = int(match.group(1))
                    if pid >= start_pid:
                        img_path.unlink()  # 物理执行删除
                        count += 1
        logger.info(f"回滚完成！成功清理 {count} 张冲突数据，数据集已恢复至健康基准状态。")

    def _get_max_existing_pid(self):
        """精准嗅探当前图库中的最大合法身份标签"""
        max_pid = 0
        extensions = ('*.jpg', '*.png', '*.tif', '*.jpeg')
        for sub_dir in self.dirs.values():
            if not sub_dir.exists(): continue
            for ext in extensions:
                for img_path in sub_dir.glob(ext):
                    match = re.match(r'^(\d+)_', img_path.name)
                    if match:
                        pid = int(match.group(1))
                        if pid > max_pid: max_pid = pid
        logger.info(f"环境扫描完毕，当前图库真实最大身份 ID 为: {max_pid}")
        return max_pid

    def get_modality_from_filename(self, file_path):
        """
        核心物理规则：从文件名的最后一个 '_' 后面精准分离模态信息
        例如：1_1024_500_8000_rgb.png -> rgb -> 光学(1)
              1_1024_500_8000_sar.png -> sar -> 雷达(2)
        """
        filename = Path(file_path).stem  # 去掉文件后缀，获取纯文件名
        parts = filename.split('_')
        
        if len(parts) > 1:
            mod_str = parts[-1].lower()  # 获取最后一段标识
            if mod_str in ['rgb', 'opt', 'optical']:
                return 'opt', 1          # 映射为光学模态标签
            elif mod_str == 'sar':
                return 'sar', 2          # 映射为雷达模态标签
                
        return 'unknown', 0

    def parse_and_crop(self):
        logger.info("=== 第一阶段：解析标注并进行跨目录多源图像抓取 (含云增强数据) ===")
        label_files = list(self.raw_data_root.rglob("*.txt"))
        
        for txt_path in label_files:
            parent_name = txt_path.parent.name
            
            # 高级路由：建立一个标注文件对多个数据增强图像文件夹的“一对多”映射
            img_parent_names = []
            if parent_name == 'labelTxt':
                img_parent_names = ['images']
            elif parent_name == 'rgb_labelTxt':
                img_parent_names = ['rgb', 'rgb_clouds']  # 核心点：同时挖掘原图与加云图
            elif parent_name == 'sar_labelTxt':
                img_parent_names = ['sar']
            else:
                continue

            for img_p_name in img_parent_names:
                img_base_dir = txt_path.parent.parent / img_p_name
                
                # 在目标文件夹中跨目录检索对应的同名图像
                img_path = None
                for ext in ['.jpg', '.png', '.tif', '.jpeg', '.JPG', '.PNG', '.TIF', '.JPEG']:
                    temp_path = img_base_dir / (txt_path.stem + ext)
                    if temp_path.exists():
                        img_path = temp_path
                        break
                
                if not img_path: continue
                
                # 严格基于文件名最后一个下划线后的文本推断模态，规避路径层级干扰
                mod_str, cam_id = self.get_modality_from_filename(img_path)
                if cam_id == 0: continue
                
                img = cv2.imread(str(img_path))
                if img is None: continue
                h_img, w_img = img.shape[:2]

                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    
                for idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 10: continue
                    
                    try:
                        # 提取8点旋转框坐标并求极值转化为高内聚水平外接矩形
                        coords = [float(x) for x in parts[:8]]
                        xs, ys = coords[0::2], coords[1::2]
                        obj_id = parts[9]
                        
                        xmin, xmax = int(min(xs)), int(max(xs))
                        ymin, ymax = int(min(ys)), int(max(ys))
                        
                        # 外扩边界填充保护目标边缘刚体特征
                        xmin = max(0, xmin - self.padding)
                        ymin = max(0, ymin - self.padding)
                        xmax = min(w_img, xmax + self.padding)
                        ymax = min(h_img, ymax + self.padding)
                        
                        if xmax - xmin < 10 or ymax - ymin < 10: continue
                        
                        crop_img = img[ymin:ymax, xmin:xmax]
                        self.all_crops[obj_id].append({
                            'image': crop_img,
                            'mod_str': mod_str,
                            'cam_id': cam_id
                        })
                    except Exception:
                        continue
                        
        # 输出数据完整性校验审计日志
        opt_count = sum(1 for crops in self.all_crops.values() for c in crops if c['cam_id'] == 1)
        sar_count = sum(1 for crops in self.all_crops.values() for c in crops if c['cam_id'] == 2)
        logger.info(f"第一阶段成功完成！从 MOS-Ship 中共抽离出 {len(self.all_crops)} 个有效实体类。")
        logger.info(f"  -> 成功捕获光学切片 (包含原图与 clouds 增强图): {opt_count} 张")
        logger.info(f"  -> 成功捕获雷达 SAR 切片: {sar_count} 张")

    def split_and_save(self):
        logger.info(f"=== 第二阶段：连续性命名映射与增量数据安全注入 ===")
        # 过滤无法构成对比三元组的单图片孤立目标
        valid_ids = [uid for uid, crops in self.all_crops.items() if len(crops) >= 2]
        
        random.seed(1949)
        random.shuffle(valid_ids)
        num_train = int(len(valid_ids) * self.train_ratio)
        train_ids, test_ids = valid_ids[:num_train], valid_ids[num_train:]
        
        logger.info(f"数据清洗流水线将以全局递增 ID 【{self.start_pid:04d}】 作为安全起点。")
        current_pid = self.start_pid
        injected_count = 0
        
        # 1. 安全注入 bounding_box_train 目录
        for uid in train_ids:
            for crop in self.all_crops[uid]:
                filename = f"{current_pid:04d}_s{crop['cam_id']}c1_{crop['mod_str']}.jpg"
                cv2.imwrite(str(self.dirs['train'] / filename), crop['image'])
                injected_count += 1
            current_pid += 1
            
        # 2. 安全注入测试目录 (实现标准的 Zero-shot 验证集ID互斥)
        for uid in test_ids:
            crops = self.all_crops[uid]
            query_candidate_idx = [i for i, c in enumerate(crops) if c['mod_str'] == self.query_modality]
            q_idx = query_candidate_idx[0] if query_candidate_idx else 0
                
            for i, crop in enumerate(crops):
                filename = f"{current_pid:04d}_s{crop['cam_id']}c1_{crop['mod_str']}.jpg"
                if i == q_idx:
                    cv2.imwrite(str(self.dirs['query'] / filename), crop['image'])
                else:
                    cv2.imwrite(str(self.dirs['gallery'] / filename), crop['image'])
                injected_count += 1
            current_pid += 1

        logger.info(f"🎉 恭喜！跨模态数据增量洗牌整合全部圆满完成。")
        logger.info(f"   本次共成功物理持久化写入文件: {injected_count} 张")
        logger.info(f"   最新的全局连续最大身份标签 ID 推进至: {current_pid - 1}")

    def execute(self):
        self.parse_and_crop()
        self.split_and_save()


if __name__ == "__main__":
    # ================= 动态路由绝对物理路径配置 =================
    # 1. 原始大场景遥感检测数据集 MOS-Ship 所在的根目录
    RAW_MOS_SHIP_PATH = "/ssd_data/lixiang_data/Datasets/Opt-SAR-ReID/MOS-Ship"
    
    # 2. 之前通过第一步清洗生成的标准细粒度数据集目标存放路径 (Merged)
    EXISTING_MERGED_PATH = "/ssd_data/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data/Merged"
    # ============================================================
    
    appender = Dynamic_ReID_Appender(
        raw_mos_path=RAW_MOS_SHIP_PATH,
        existing_reid_path=EXISTING_MERGED_PATH,
        rollback_pid=771,      # 指定回滚起点，自动深度擦除该数字及之后的所有遗留图片，防止重复堆叠
        train_ratio=0.5,
        query_modality='sar',
        padding=10
    )
    appender.execute()