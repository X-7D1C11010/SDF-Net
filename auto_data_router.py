# -*- coding: utf-8 -*-
import os
import shutil
import glob
import re
import logging
from collections import defaultdict
from pathlib import Path

# ================= 配置日志 =================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AutoDataRouter:
    def __init__(self, raw_root, output_root):
        self.raw_root = Path(raw_root)
        self.out_root = Path(output_root)
        
        # 定义并创建输出目录拓扑
        self.dirs = {
            'pretrain_opt': self.out_root / 'Pretrain_Data' / 'opt',
            'pretrain_sar': self.out_root / 'Pretrain_Data' / 'sar',
            'reid_train': self.out_root / 'FineGrained_ReID' / 'bounding_box_train',
            'reid_query': self.out_root / 'FineGrained_ReID' / 'query',
            'reid_gallery': self.out_root / 'FineGrained_ReID' / 'bounding_box_test',
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        # 全局计数器与状态记录
        self.global_pid = 1  # 细粒度ReID的全局连续ID
        self.global_pretrain_id = 1 # 预训练Patch的全局连续ID
        self.report = {
            'pretrain': defaultdict(int),
            'fine_grained': defaultdict(int),
            'unsuitable': defaultdict(str)
        }

    def safe_copy(self, src, dst):
        """安全拷贝，包含基本的错误处理"""
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.error(f"拷贝文件失败: {src} -> {dst} | Error: {str(e)}")
            return False

    def process_qxs_saropt(self):
        """处理 QXS-SAROPT (同名配对的预训练数据)"""
        dataset_path = self.raw_root / "QXS-SAROPT" / "QXSLAB_SAROPT"
        if not dataset_path.exists(): return
        
        opt_dir = dataset_path / "opt_256_oc_0.2"
        sar_dir = dataset_path / "sar_256_oc_0.2"
        
        if opt_dir.exists() and sar_dir.exists():
            opt_files = {f.name for f in opt_dir.glob("*.*")}
            sar_files = {f.name for f in sar_dir.glob("*.*")}
            common_files = opt_files.intersection(sar_files)
            
            for fname in common_files:
                new_name = f"patch_{self.global_pretrain_id:06d}"
                ext = Path(fname).suffix
                
                if self.safe_copy(opt_dir / fname, self.dirs['pretrain_opt'] / f"{new_name}_opt{ext}") and \
                   self.safe_copy(sar_dir / fname, self.dirs['pretrain_sar'] / f"{new_name}_sar{ext}"):
                    self.global_pretrain_id += 1
                    self.report['pretrain']['QXS-SAROPT'] += 1
        logger.info("QXS-SAROPT 处理完成 (划分至预训练集)")

    def process_3mos(self):
        """处理 3MOS (数字ID配对的预训练数据)"""
        dataset_path = self.raw_root / "3MOS"
        if not dataset_path.exists(): return
        
        for sensor in ["GF3", "Radarsat", "RCM", "ALOS"]:
            sensor_path = dataset_path / sensor
            opt_dir, sar_dir = sensor_path / "opt", sensor_path / "sar"
            
            if opt_dir.exists() and sar_dir.exists():
                # 提取数字进行匹配 (如 sar_690.jpg 和 opt_690.jpg)
                opt_dict = {re.search(r'\d+', f.name).group(): f for f in opt_dir.rglob("*.*") if re.search(r'\d+', f.name)}
                sar_dict = {re.search(r'\d+', f.name).group(): f for f in sar_dir.rglob("*.*") if re.search(r'\d+', f.name)}
                
                common_ids = set(opt_dict.keys()).intersection(set(sar_dict.keys()))
                for cid in common_ids:
                    new_name = f"patch_{self.global_pretrain_id:06d}"
                    opt_ext = opt_dict[cid].suffix
                    sar_ext = sar_dict[cid].suffix
                    
                    if self.safe_copy(opt_dict[cid], self.dirs['pretrain_opt'] / f"{new_name}_opt{opt_ext}") and \
                       self.safe_copy(sar_dict[cid], self.dirs['pretrain_sar'] / f"{new_name}_sar{sar_ext}"):
                        self.global_pretrain_id += 1
                        self.report['pretrain'][f'3MOS-{sensor}'] += 1
        logger.info("3MOS 处理完成 (划分至预训练集)")

    def process_hoss_reid(self):
        """处理 HOSS-ReID (含细粒度类别ID的数据)"""
        dataset_path = self.raw_root / "HOSS-ReID"
        if not dataset_path.exists(): return

        # 1. 处理已经分好类的 OptiSar_Pair
        pair_dir = dataset_path / "OptiSar_Pair"
        if pair_dir.exists():
            for class_folder in pair_dir.iterdir():
                if class_folder.is_dir():
                    imgs = list(class_folder.glob("*.*"))
                    if not imgs: continue
                    
                    for img in imgs:
                        modality = "RGB" if "RGB" in img.name else "SAR"
                        camid = 1 if modality == "RGB" else 2
                        new_name = f"{self.global_pid:04d}_s{camid}c1_{modality}{img.suffix}"
                        if self.safe_copy(img, self.dirs['reid_train'] / new_name):
                            self.report['fine_grained']['HOSS-ReID (OptiSar_Pair)'] += 1
                    self.global_pid += 1 # 每个文件夹分配一个新的全局连续ID

        # 2. 处理原版 HOSS
        hoss_dir = dataset_path / "HOSS"
        if hoss_dir.exists():
            for subset, target_dir in [("bounding_box_train", self.dirs['reid_train']),
                                       ("query", self.dirs['reid_query']),
                                       ("bounding_box_test", self.dirs['reid_gallery'])]:
                src_dir = hoss_dir / subset
                if not src_dir.exists(): continue
                
                # 建立原ID到新全局连续ID的映射字典
                local_to_global = {}
                for img in src_dir.glob("*.*"):
                    match = re.match(r'^(-?\d+)_', img.name)
                    if match:
                        old_id = match.group(1)
                        if old_id not in local_to_global:
                            local_to_global[old_id] = self.global_pid
                            self.global_pid += 1
                        
                        # 替换前缀ID，保留原后缀
                        new_name = img.name.replace(f"{old_id}_", f"{local_to_global[old_id]:04d}_", 1)
                        if self.safe_copy(img, target_dir / new_name):
                            self.report['fine_grained'][f'HOSS-ReID ({subset})'] += 1
        logger.info("HOSS-ReID 处理完成 (划分至细粒度训练集)")

    def mark_unsuitable_datasets(self):
        """标记不适合跨模态配准/ReID任务的数据集及原因"""
        unsuitable_rules = {
            "MOS-Ship": "单一目标检测数据集，主要提供边界框(Bounding Box)坐标，缺乏严格对应的跨模态图像对。",
            "Multi-Resolution-SAR-dataset": "单模态SAR数据集，主要用于分辨率增强或单模态识别，无光学配对数据。",
            "OSdataset": "主要用于船舶检测和语义分割任务，未经跨模态图像级空间配准。",
            "OsEval": "目标检测评估基准数据集，缺乏成对的跨模态训练补丁(Patch)。",
            "OSDataset2.0": "虽然包含部分切片，但其核心聚焦于场景级(Scene-level)检测，缺乏细粒度ID且配对关系不严格。"
        }
        
        for ds_name, reason in unsuitable_rules.items():
            if (self.raw_root / ds_name).exists():
                self.report['unsuitable'][ds_name] = reason
                logger.warning(f"跳过数据集: {ds_name} - 原因: {reason}")

    def execute(self):
        logger.info(">>> 开始自动化数据处理与洗牌流程 <<<")
        self.process_qxs_saropt()
        self.process_3mos()  # <--- 改成小写
        self.process_hoss_reid()
        self.mark_unsuitable_datasets()
        self.generate_report()

    def generate_report(self):
        """生成并打印终端报告"""
        print("\n" + "="*60)
        print("                 🏁 数据处理与分类最终报告")
        print("="*60)
        
        print("\n[一] 预训练数据集 (Pre-training Data) - 用于跨越模态鸿沟")
        print(f"输出路径 (OPT): {self.dirs['pretrain_opt']}")
        print(f"输出路径 (SAR): {self.dirs['pretrain_sar']}")
        print("包含的数据集及提取图像对数量:")
        for ds, count in self.report['pretrain'].items():
            print(f"  - {ds}: {count} 对")
            
        print("\n[二] 细粒度训练数据集 (Fine-Grained ReID Data) - 用于学习身份特征")
        print(f"输出路径 (Train): {self.dirs['reid_train']}")
        print(f"输出路径 (Query): {self.dirs['reid_query']}")
        print(f"输出路径 (Gallery): {self.dirs['reid_gallery']}")
        print("包含的数据集及提取图像总数:")
        for ds, count in self.report['fine_grained'].items():
            print(f"  - {ds}: {count} 张")
        print(f"  * 累计生成全局连续身份类别(ID)数量: {self.global_pid - 1} 个")
            
        print("\n[三] 已隔离/不适合当前任务的数据集 (Unsuitable Data)")
        for ds, reason in self.report['unsuitable'].items():
            print(f"  - 🚫 {ds}")
            print(f"    原因: {reason}")
            
        print("\n" + "="*60)
        print("提示: 后续训练时，请分别将配置文件中的 ROOT_DIR 指向上述 [一] 或 [二] 的根目录。")
        print("="*60 + "\n")

if __name__ == "__main__":
    # ================= 用户配置区 =================
    # 原始数据集的总目录（包含3MOS, HOSS-ReID等文件夹的路径）
    RAW_DATA_ROOT = "/ssd_data/lixiang_data/Datasets/Opt-SAR-ReID" 
    
    # 清洗后的数据要保存的全新目标路径 (建议新建一个干净的文件夹)
    OUTPUT_ROOT = "/ssd_data/lixiang_data/Datasets/Opt-SAR-ReID/Cleaned_SDFNet_Data"
    # ==============================================
    
    router = AutoDataRouter(RAW_DATA_ROOT, OUTPUT_ROOT)
    router.execute()