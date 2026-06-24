# -*- coding: utf-8 -*-
import os
from pathlib import Path

def batch_rename_modalities(target_dir):
    base_dir = Path(target_dir)
    # ReID 的三个标准子目录
    sub_dirs = ['bounding_box_train', 'query', 'bounding_box_test']
    
    renamed_count = 0
    
    print(f"开始扫描目录: {base_dir}")
    
    for sub in sub_dirs:
        folder_path = base_dir / sub
        if not folder_path.exists():
            continue
            
        # 遍历该目录下的所有图片
        for file_path in folder_path.glob('*.*'):
            name = file_path.stem  # 不带后缀的文件名
            ext = file_path.suffix # 文件后缀 (如 .jpg)
            
            parts = name.split('_')
            
            if len(parts) > 1:
                old_mod_str = parts[-1]
                lower_mod_str = old_mod_str.lower()
                new_mod_str = None
                
                # 判断并映射为统一的大写标识
                if lower_mod_str in ['opt', 'rgb', 'optical', 'images']:
                    new_mod_str = 'RGB'
                elif lower_mod_str == 'sar':
                    new_mod_str = 'SAR'
                    
                # 如果后缀不规范（包含小写等），则执行重命名
                if new_mod_str and old_mod_str != new_mod_str:
                    parts[-1] = new_mod_str
                    new_name = "_".join(parts) + ext
                    new_file_path = folder_path / new_name
                    
                    # 物理重命名
                    file_path.rename(new_file_path)
                    renamed_count += 1
                    
    print(f"🎉 批量重命名完成！")
    print(f"共扫描并修正了 {renamed_count} 个文件的模态名称。")

if __name__ == '__main__':
    # 指向你已经整合好的 Merged 数据集目录
    TARGET_DIR = "/ssd_data/lixiang_data/Datasets/Opt-SAR-ReID/data"
    
    batch_rename_modalities(TARGET_DIR)