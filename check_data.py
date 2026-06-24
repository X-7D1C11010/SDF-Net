import os

def tree_view(startpath, max_depth=2):
    print(f"\n" + "="*50)
    print(f"=== 扫描目标: {startpath}")
    if not os.path.exists(startpath):
        print(" [X] 路径不存在，已跳过。")
        return

    for root, dirs, files in os.walk(startpath):
        # 计算当前目录的深度
        depth = root[len(startpath):].count(os.sep)
        if depth > max_depth:
            del dirs[:]  # 超过指定深度就不往下扫了，防止刷屏
            continue
        
        indent = ' ' * 4 * depth
        folder_name = os.path.basename(root) if root != startpath else os.path.basename(os.path.normpath(startpath))
        print(f"{indent}📁 {folder_name}/  (包含子文件夹: {len(dirs)} 个, 文件: {len(files)} 个)")
        
        # 挑几张图片名字看看规律
        imgs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        if imgs:
            print(f"{indent}    📄 样本文件: {imgs[:3]}")

if __name__ == "__main__":
    # ！！！请核对并修改这里的父目录路径！！！
    # 假设你的这 8 个原始数据集都放在这个目录下
    RAW_DATA_ROOT = "/ssd_data/lixiang_data/Datasets/Opt-SAR-ReID" 
    
    # 你提到的原始数据集名称（如果名字有出入，请直接修改列表里的字符串）
    DATASETS = [
        "3MOS", 
        "HOSS-ReID", 
        "MOS-Ship", 
        "Multi-Resolution-SAR-dataset",
        "OSdataset", 
        "OSDataset2.0", 
        "OsEval", 
        "QXS-SAROPT"
    ]
    
    for ds in DATASETS:
        ds_path = os.path.join(RAW_DATA_ROOT, ds)
        tree_view(ds_path, max_depth=2)