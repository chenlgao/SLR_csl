import os
import torch

class Config:
    # 路径配置
    raw_data_root = r"D:\C\csl\color_video_25000"
    npz_dir = "D:\ctcn_2/npz_dataset"
    model_save_dir = 'D:\ctcn_2\models'
    dictionary_path = "dictionary.txt"
    
    # 模型参数
    model_name = 'resnet18'
    max_sequence_length = 170
    num_classes = 99
    
    # 训练参数
    batch_size = 128
    epochs = 300
    learning_rate = 0.00001
    patience = 10
    min_delta = 0.001
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()