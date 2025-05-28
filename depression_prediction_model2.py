import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡问题"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ResidualBlock(nn.Module):
    """残差块组件"""
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(input_dim)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.batch_norm1(self.fc1(x)))
        out = self.dropout(out)
        out = self.batch_norm2(self.fc2(out))
        out += residual  # 残差连接
        return F.relu(out)

class LSTMFeatureExtractor(nn.Module):
    """LSTM特征提取器"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout_rate=0.2):
        super(LSTMFeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = 4
        features_per_step = x.size(1) // seq_len
        
        if x.size(1) % seq_len != 0:
            padding_size = seq_len - (x.size(1) % seq_len)
            x = F.pad(x, (0, padding_size))
            features_per_step = x.size(1) // seq_len
        
        x_seq = x.view(batch_size, seq_len, features_per_step)
        lstm_out, (hidden, cell) = self.lstm(x_seq)
        last_output = lstm_out[:, -1, :]
        return self.dropout(last_output)

class AttentionLayer(nn.Module):
    """注意力机制层"""
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        attention_weights = self.attention(x)
        attended_output = torch.sum(attention_weights * x, dim=1)
        return attended_output

class HybridDepressionPredictor(nn.Module):
    """优化的混合神经网络：提高召回率"""
    def __init__(self, input_dim, lstm_hidden=64, res_hidden=128, fc_hidden=256, dropout_rate=0.2):
        super(HybridDepressionPredictor, self).__init__()
        
        # 输入特征预处理层
        self.input_fc = nn.Linear(input_dim, 64)
        self.input_bn = nn.BatchNorm1d(64)
        
        # LSTM分支
        self.lstm_branch = LSTMFeatureExtractor(
            input_dim=16,
            hidden_dim=lstm_hidden,
            num_layers=3,  # 增加层数
            dropout_rate=dropout_rate
        )
        
        # 残差网络分支
        self.res_branch = nn.Sequential(
            nn.Linear(64, res_hidden),
            nn.BatchNorm1d(res_hidden),
            nn.ReLU(),
            ResidualBlock(res_hidden, res_hidden // 2, dropout_rate),
            ResidualBlock(res_hidden, res_hidden // 2, dropout_rate),
            ResidualBlock(res_hidden, res_hidden // 2, dropout_rate),
            ResidualBlock(res_hidden, res_hidden // 2, dropout_rate)  # 增加残差块
        )
        
        # 全连接分支
        self.fc_branch = nn.Sequential(
            nn.Linear(64, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.BatchNorm1d(fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden // 2, fc_hidden // 4),  # 增加层数
            nn.BatchNorm1d(fc_hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 特征融合层
        fusion_input_dim = lstm_hidden * 2 + res_hidden + fc_hidden // 4
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),  # 增加容量
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 最终分类层 - 针对召回率优化
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # 减少dropout以保留更多信息
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x_processed = F.relu(self.input_bn(self.input_fc(x)))
        
        lstm_features = self.lstm_branch(x_processed)
        res_features = self.res_branch(x_processed)
        fc_features = self.fc_branch(x_processed)
        
        fused_features = torch.cat([lstm_features, res_features, fc_features], dim=1)
        fused_output = self.fusion_layer(fused_features)
        
        output = self.classifier(fused_output)
        return output.squeeze()

class DepressionDataProcessor:
    """数据预处理类"""
    def __init__(self, use_smote=True):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.use_smote = use_smote
        
    def preprocess_data(self, df):
        """预处理数据"""
        df_processed = df.copy()
        
        categorical_features = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Degree', 
                              'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
        
        for feature in categorical_features:
            if feature in df_processed.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    df_processed[feature] = self.label_encoders[feature].fit_transform(df_processed[feature].astype(str))
                else:
                    df_processed[feature] = self.label_encoders[feature].transform(df_processed[feature].astype(str))
        
        X = df_processed.drop('Depression', axis=1)
        y = df_processed['Depression']
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 使用SMOTE进行过采样以平衡数据集
        if self.use_smote:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_scaled, y = smote.fit_resample(X_scaled, y.values)
            print(f"SMOTE后的数据分布: {np.bincount(y)}")
        
        return X_scaled, y if self.use_smote else y.values

def calculate_class_weights(y):
    """计算类别权重"""
    unique_classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    class_weights = total_samples / (len(unique_classes) * counts)
    return torch.FloatTensor(class_weights)

def get_optimal_threshold(model, data_loader, device):
    """寻找最优阈值以最大化召回率"""
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    
    best_threshold = 0.5
    best_recall = 0
    
    # 测试不同阈值
    for threshold in np.arange(0.1, 0.9, 0.05):
        predictions = (all_outputs > threshold).astype(int)
        recall = recall_score(all_targets, predictions)
        if recall > best_recall:
            best_recall = recall
            best_threshold = threshold
    
    return best_threshold, best_recall

def train_model_optimized():
    """优化的训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    df = pd.read_csv('d:\\人工智能课程设计\\copy.csv')
    print(f"原始数据分布: {df['Depression'].value_counts().to_dict()}")
    
    # 数据预处理
    processor = DepressionDataProcessor(use_smote=True)
    X, y = processor.preprocess_data(df)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 计算类别权重
    class_weights = calculate_class_weights(y_train)
    pos_weight = class_weights[1] / class_weights[0]
    print(f"正样本权重: {pos_weight:.2f}")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 减小batch size
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    input_dim = X_train.shape[1]
    model = HybridDepressionPredictor(input_dim, dropout_rate=0.15).to(device)
    
    # 使用Focal Loss和加权BCE Loss的组合
    focal_loss = FocalLoss(alpha=2, gamma=2)
    bce_loss = nn.BCELoss()
    
    # 优化器设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    # 训练循环
    num_epochs = 150
    best_recall = 0.0
    best_threshold = 0.5
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # 组合损失函数
            loss1 = focal_loss(outputs, batch_y)
            loss2 = bce_loss(outputs, batch_y)
            loss = 0.7 * loss1 + 0.3 * loss2  # 重点关注focal loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # 验证
        if epoch % 5 == 0:
            model.eval()
            all_outputs = []
            all_targets = []
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    val_loss += bce_loss(outputs, batch_y).item()
                    all_outputs.extend(outputs.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            # 寻找最优阈值
            threshold, recall = get_optimal_threshold(model, test_loader, device)
            
            # 使用最优阈值计算指标
            predictions = (np.array(all_outputs) > threshold).astype(int)
            precision = precision_score(all_targets, predictions)
            f1 = f1_score(all_targets, predictions)
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            
            print(f'Epoch [{epoch}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Threshold: {threshold:.3f}, '
                  f'Recall: {recall:.4f}, '
                  f'Precision: {precision:.4f}, '
                  f'F1: {f1:.4f}')
            
            # 保存最佳模型（基于召回率）
            if recall > best_recall:
                best_recall = recall
                best_threshold = threshold
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'threshold': threshold,
                    'recall': recall,
                    'precision': precision,
                    'f1': f1
                }, 'best_recall_depression_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"早停：{patience}个epoch没有改善")
                break
    
    print(f'最佳召回率: {best_recall:.4f}, 最优阈值: {best_threshold:.3f}')
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor).cpu().numpy()
        test_predictions = (test_outputs > best_threshold).astype(int)
        
        print("\n=== 最终测试结果 ===")
        print(f"最优阈值: {best_threshold:.3f}")
        print("\n分类报告:")
        print(classification_report(y_test, test_predictions, target_names=['无抑郁', '有抑郁']))
        print("\n混淆矩阵:")
        print(confusion_matrix(y_test, test_predictions))
    
    return model, processor, best_threshold

if __name__ == "__main__":
    model, processor, threshold = train_model_optimized()