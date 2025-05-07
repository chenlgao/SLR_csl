# modeltrain.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from CNN_LSTM import FeatureExtractor, MultiModalCNNTransformerModel
from config import config
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.transforms.functional import convert_image_dtype

# 创建特征分析目录
os.makedirs("feature_analysis", exist_ok=True)

class NPZDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.data_dir = os.path.join(config.npz_dir, mode)
        
        # 加载标签字典
        self.index_to_label = {}
        with open(config.dictionary_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                code, label = line.strip().split('\t')
                self.index_to_label[idx] = label

        # 加载数据
        self.file_list = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])
        features, labels, lengths = [], [], []
        
        for file in self.file_list:
            with np.load(os.path.join(self.data_dir, file)) as data:
                features.append(data['features'].astype(np.float32))
                labels.append(data['label'].item())
                lengths.append(data['length'].astype(np.int64))
        
        self.features = np.stack(features)
        self.labels = np.stack(labels)
        self.lengths = np.stack(lengths)

    def __len__(self): 
        return len(self.file_list)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long)
        )

def custom_collate(batch):
    """修正后的批处理函数"""
    # 按序列长度排序
    sorted_batch = sorted(batch, key=lambda x: x[2], reverse=True)
    features = [item[0] for item in sorted_batch]
    labels = [item[1] for item in sorted_batch]
    lengths = [item[2] for item in sorted_batch]
    
    # 填充特征序列
    padded_features = torch.nn.utils.rnn.pad_sequence(
        features,
        batch_first=True,
        padding_value=0.0
    )
    return padded_features, torch.stack(labels), torch.stack(lengths)

def visualize_features(features, labels, epoch):
    """特征可视化（PCA + t-SNE）"""
    # PCA降维
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.title(f'PCA - Epoch {epoch+1}')
    plt.colorbar()

    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(features)
    
    plt.subplot(1, 2, 2)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.title(f't-SNE - Epoch {epoch+1}')
    plt.colorbar()
    
    plt.savefig(f'feature_analysis/epoch_{epoch+1}.png')
    plt.close()

def train():
    # 初始化数据集
    train_dataset = NPZDataset('train')
    val_dataset = NPZDataset('test')

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=custom_collate,
        pin_memory=True
    )

    # 初始化模型
    model = MultiModalCNNTransformerModel(
        feature_dim=512 + 126, 
        num_classes=config.num_classes, 
        heads=2  # 设置为2或其他638的因数
    ).to(config.device)  # 更新特征维度
    feature_extractor = FeatureExtractor().to(config.device)
    
    # 冻结特征提取器（NPZ数据已包含预提取特征）
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # 优化器配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.patience)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 训练循环
    best_acc = 0.0
    for epoch in range(config.epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        
        # 训练阶段
        for features, labels, lengths in train_loader:
            features = features.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(features, lengths)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算训练指标
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_loss += loss.item() * features.size(0)

        # 验证阶段
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels, lengths in val_loader:
                features = features.to(config.device)
                labels = labels.to(config.device)
                
                outputs = model(features, lengths)
                val_loss += criterion(outputs, labels).item() * features.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                # 新增：收集特征和标签（假设需保存分类前的特征）
                all_features.append(outputs.detach().cpu().numpy())  # 保存logits
                all_labels.append(labels.detach().cpu().numpy()) 

        # 特征可视化（每5个epoch执行一次）
        if (epoch + 1) % 5 == 0:
            features_concatenated = np.concatenate(all_features, axis=0)
            labels_concatenated = np.concatenate(all_labels, axis=0)
            visualize_features(features_concatenated, labels_concatenated, epoch)

        # 计算指标
        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct / len(train_dataset)
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_acc + config.min_delta:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'index_to_label': train_dataset.index_to_label,
                'max_sequence_length': config.max_sequence_length
            }, os.path.join(config.model_save_dir, 'best_model.pth'))

        # 打印日志
        print(f"Epoch {epoch+1:03d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | "
              f"Best Acc: {best_acc:.2%}")

if __name__ == '__main__':
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs("feature_analysis", exist_ok=True)
    
    print("\n=== 训练配置 ===")
    print(f"设备: {config.device}")
    print(f"特征提取器: {config.model_name} (使用预提取特征)")
    print(f"最大序列长度: {config.max_sequence_length}")
    print(f"类别数量: {config.num_classes}\n")
    
    train()