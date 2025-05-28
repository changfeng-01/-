import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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
        # 将输入重塑为序列格式 (batch_size, seq_len, features)
        batch_size = x.size(0)
        # 创建时间序列：将特征分组作为时间步
        seq_len = 4  # 将特征分为4个时间步
        features_per_step = x.size(1) // seq_len
        
        # 如果特征数不能被seq_len整除，进行填充
        if x.size(1) % seq_len != 0:
            padding_size = seq_len - (x.size(1) % seq_len)
            x = F.pad(x, (0, padding_size))
            features_per_step = x.size(1) // seq_len
        
        x_seq = x.view(batch_size, seq_len, features_per_step)
        
        lstm_out, (hidden, cell) = self.lstm(x_seq)
        
        # 使用最后一个时间步的输出和隐藏状态
        # 双向LSTM的输出维度是 hidden_dim * 2
        last_output = lstm_out[:, -1, :]
        return self.dropout(last_output)

class HybridDepressionPredictor(nn.Module):
    """混合神经网络：结合LSTM、ResNet和全连接层"""
    def __init__(self, input_dim, lstm_hidden=64, res_hidden=128, fc_hidden=256, dropout_rate=0.3):
        super(HybridDepressionPredictor, self).__init__()
        
        # 输入特征预处理层
        self.input_fc = nn.Linear(input_dim, 64)
        self.input_bn = nn.BatchNorm1d(64)
        
        # LSTM分支 - 用于捕获特征间的序列依赖关系
        self.lstm_branch = LSTMFeatureExtractor(
            input_dim=16,  # 每个时间步的特征数
            hidden_dim=lstm_hidden,
            num_layers=2,
            dropout_rate=dropout_rate
        )
        
        # 残差网络分支 - 用于深层特征学习
        self.res_branch = nn.Sequential(
            nn.Linear(64, res_hidden),
            nn.BatchNorm1d(res_hidden),
            nn.ReLU(),
            ResidualBlock(res_hidden, res_hidden // 2, dropout_rate),
            ResidualBlock(res_hidden, res_hidden // 2, dropout_rate),
            ResidualBlock(res_hidden, res_hidden // 2, dropout_rate)
        )
        
        # 全连接分支 - 用于直接特征映射
        self.fc_branch = nn.Sequential(
            nn.Linear(64, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.BatchNorm1d(fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 特征融合层
        fusion_input_dim = lstm_hidden * 2 + res_hidden + fc_hidden // 2  # LSTM是双向的
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 输入预处理
        x_processed = F.relu(self.input_bn(self.input_fc(x)))
        
        # LSTM分支
        lstm_features = self.lstm_branch(x_processed)
        
        # 残差网络分支
        res_features = self.res_branch(x_processed)
        
        # 全连接分支
        fc_features = self.fc_branch(x_processed)
        
        # 特征融合
        fused_features = torch.cat([lstm_features, res_features, fc_features], dim=1)
        fused_output = self.fusion_layer(fused_features)
        
        # 最终预测
        output = self.classifier(fused_output)
        return output.squeeze()

class DepressionDataProcessor:
    """数据预处理类"""
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df):
        """预处理数据"""
        df_processed = df.copy()
        
        # 处理分类特征
        categorical_features = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Degree', 
                              'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
        
        for feature in categorical_features:
            if feature in df_processed.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    df_processed[feature] = self.label_encoders[feature].fit_transform(df_processed[feature].astype(str))
                else:
                    df_processed[feature] = self.label_encoders[feature].transform(df_processed[feature].astype(str))
        
        # 分离特征和标签
        X = df_processed.drop('Depression', axis=1)
        y = df_processed['Depression']
        
        # 标准化数值特征
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values

def train_model():
    """训练模型"""
    # 加载数据
    df = pd.read_csv('d:\\人工智能课程设计\\copy.csv')
    
    # 数据预处理
    processor = DepressionDataProcessor()
    X, y = processor.preprocess_data(df)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    input_dim = X_train.shape[1]
    model = HybridDepressionPredictor(input_dim)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 训练循环
    num_epochs = 100
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        scheduler.step(avg_val_loss)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_depression_model.pth')
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Accuracy: {accuracy:.2f}%')
    
    print(f'Best Accuracy: {best_accuracy:.2f}%')
    return model, processor

if __name__ == "__main__":
    model, processor = train_model()