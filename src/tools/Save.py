import os
import glob
import torch
from Config import save_dir

def save_rolling_checkpoint(
    model,
    metric_value,
    metric_name,
    max_keep=3,  # 每个指标最多保留的权重数量
):
    """滚动覆盖式保存指标最佳权重（分指标存储）"""
    # 确保目录存在（按指标分类）
    metric_dir = os.path.join(save_dir)
    os.makedirs(metric_dir, exist_ok=True)
    
    # 生成标准化文件名（包含epoch时间戳避免冲突）
    filename = f"{metric_name}_{metric_value:.6f}.pth"
    filepath = os.path.join(metric_dir, filename)
    
    # 保存当前模型
    torch.save(model.state_dict(), filepath)
    print(f"Saved {metric_name} checkpoint: {filepath}")
    
    # 获取该指标目录下所有文件
    all_files = glob.glob(os.path.join(metric_dir, f"{metric_name}_*.pth"))
    
    # 按指标值排序（降序）
    all_files.sort(
        key=lambda x: float(os.path.basename(x).split("_")[1].split(".pth")[0]),
        reverse=True
    )
    
    # 删除超出max_keep的旧文件
    for old_file in all_files[max_keep:]:
        os.remove(old_file)
        print(f"Deleted old {metric_name} checkpoint: {old_file}")

def update_and_save_best_metric(current_value, best_value, model, name, log_file):
    if current_value > best_value:
        best_value = current_value
        print(f'Best_{name} : {current_value:.6f}')
        save_rolling_checkpoint(model, current_value, name)
        log_file.write(f'Best_{name}: {current_value:.6f}\n')
    return best_value
