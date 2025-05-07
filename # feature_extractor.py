# feature_extractor.py
import os
import numpy as np
import torch
from tqdm import tqdm
from CNN_LSTM import FeatureExtractor, MultiModalCNNTransformerModel
from yuchuli import load_frames_and_keypoints_from_directory  # 修改导入函数名
from config import config  # 导入配置文件

# 创建输出目录
os.makedirs(os.path.join(config.npz_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(config.npz_dir, "test"), exist_ok=True)

# 加载标签字典并生成映射
label_to_index = {}  # 类别编号 -> 连续索引（0~3）
valid_codes = set()  # 有效类别编号集合（如000001）
with open(config.dictionary_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        code, _ = line.strip().split('\t')
        valid_codes.add(code)
        label_to_index[code] = idx  # 建立映射关系（000001 -> 0, 000002 -> 1等）

# 初始化特征提取器
feature_extractor = FeatureExtractor(model_name=config.model_name).to(config.device)
feature_extractor.eval()

def process_sequence(seq_dir):
    """处理单个序列目录（如P01_01_01_4）"""
    frames, keypoints = load_frames_and_keypoints_from_directory(seq_dir)  # 修改函数调用
    if frames is None or len(frames) < 5 or keypoints is None:
        print(f"跳过无效序列：{seq_dir}（帧数不足）")
        return None
    
    # 转换为Tensor并提取特征
    frame_tensors = torch.tensor(frames).permute(0, 3, 1, 2).float().to(config.device)
    keypoints_tensors = torch.tensor(keypoints).float().to(config.device)
    
    with torch.no_grad():
        # 提取视频帧特征
        visual_features = feature_extractor(frame_tensors)
        # 融合关键点特征
        fused_features = torch.cat([visual_features, keypoints_tensors], dim=1)
    
    # 截断/填充序列
    seq_length = min(len(fused_features), config.max_sequence_length)
    if len(fused_features) > config.max_sequence_length:
        fused_features = fused_features[:config.max_sequence_length]
    else:
        padding = torch.zeros((config.max_sequence_length - seq_length, fused_features.shape[1])).to(config.device)
        fused_features = torch.cat([fused_features, padding])
    
    return fused_features.cpu().numpy(), seq_length

# 收集所有样本信息
all_samples = []
print("正在扫描有效数据...")

# 遍历类别目录（如000001）
for class_code in os.listdir(config.raw_data_root):
    class_path = os.path.join(config.raw_data_root, class_code)
    
    # 跳过非目录或无效类别
    if not os.path.isdir(class_path) or class_code not in valid_codes:
        continue

    # 遍历序列目录（如P01_01_01_4）
    sequence_dirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
    for seq_dir in tqdm(sequence_dirs, desc=f"处理类别 {class_code}"):
        seq_path = os.path.join(class_path, seq_dir)
        result = process_sequence(seq_path)
        if not result:
            continue
        
        features, length = result
        all_samples.append({
            'features': features,
            'label': label_to_index[class_code],  # 使用连续索引标签（0~3）
            'length': length,
            'class_code': class_code,  # 类别编号（字符串格式）
            'sequence_id': seq_dir    # 完整序列ID（如P01_01_01_4）
        })

# 划分数据集
from sklearn.model_selection import train_test_split

indices = np.arange(len(all_samples))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=[s['label'] for s in all_samples],
    random_state=42
)

# 保存数据集（保持原始命名格式）
def save_samples(indices, mode):
    for idx in tqdm(indices, desc=f"保存{mode}集"):
        sample = all_samples[idx]
        # 生成文件名：类别编号_原始序列ID.npz（如000001_P01_01_01_4.npz）
        filename = f"{sample['class_code']}_{sample['sequence_id']}.npz"
        np.savez(
            os.path.join(config.npz_dir, mode, filename),
            features=sample['features'],
            label=sample['label'],
            length=sample['length']
        )

save_samples(train_idx, "train")
save_samples(test_idx, "test")

print(f"特征生成完成！共处理 {len(all_samples)} 个序列")
print(f"输出文件示例：000001_P01_01_01_4.npz")