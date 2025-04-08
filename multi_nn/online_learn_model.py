import torch 
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from model import AnomalyDetector,ThresholdOptimizer
from data_preset import SlidingWindowDataset,create_data_loaders
class OnlineLearningSystem:
    def __init__(self, initial_threshold=0.5, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.detector = AnomalyDetector().to(self.device)
        self.optimizer = ThresholdOptimizer().to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        self.detector_optim = torch.optim.Adam(self.detector.parameters(), lr=lr)
        self.threshold_optim = torch.optim.Adam(self.optimizer.parameters(), lr=lr)
        
        self.threshold = initial_threshold
        self.best_f1 = 0.0
    
    def train_detector(self, train_loader, epochs=10):
        self.detector.train()
        for epoch in range(epochs):
            for signals, labels in train_loader:
                signals, labels = signals.to(self.device), labels.to(self.device)
                
                self.detector_optim.zero_grad()
                
                outputs, _ = self.detector(signals)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.detector_optim.step()
    
    def evaluate(self, test_loader):
        self.detector.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for signals, labels in test_loader:
                signals, labels = signals.to(self.device), labels.to(self.device)
                outputs, _ = self.detector(signals)
                
                preds = (outputs > self.threshold).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        
        return f1, precision, recall
    
    def online_update(self, new_signals, new_labels):
        # 转换为适合模型输入的格式
        dataset = SlidingWindowDataset(new_signals, labels=new_labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.detector.train()
        self.optimizer.train()
        
        all_features = []
        all_metrics = []
        all_labels = []
        
        # 第一步: 收集特征和当前性能
        with torch.no_grad():
            for signals, labels in loader:
                signals, labels = signals.to(self.device), labels.to(self.device)
                outputs, features = self.detector(signals)
                
                preds = (outputs > self.threshold).float()
                all_features.append(features)
                all_labels.append(labels)
        
        # 计算当前性能指标
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        with torch.no_grad():
            outputs, _ = self.detector(signals)
            preds = (outputs > self.threshold).float()
            
            f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())
            precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy())
            recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy())
            
            metrics = torch.FloatTensor([f1, precision, recall]).unsqueeze(0).to(self.device)
            metrics = metrics.repeat(features.size(0), 1)
        
        # 第二步: 优化阈值
        self.threshold_optim.zero_grad()
        
        # 理想情况下，我们希望阈值能最大化F1分数
        # 这里我们使用当前F1作为目标，让优化器学习如何调整阈值以达到更好的F1
        target_f1 = torch.FloatTensor([min(1.0, f1 + 0.05)]).to(self.device)  # 目标是比当前高5%
        
        # 预测新阈值
        new_thresholds = self.optimizer(features, metrics)
        
        # 计算损失 - 我们想让预测的阈值能提高F1分数
        # 这里简化处理，实际可以更复杂
        loss = self.mse_loss(new_thresholds.mean(), target_f1)
        
        loss.backward()
        self.threshold_optim.step()
        
        # 更新全局阈值(平滑更新)
        with torch.no_grad():
            suggested_threshold = new_thresholds.mean().item()
            self.threshold = 0.9 * self.threshold + 0.1 * suggested_threshold
        
        # 可选: 微调检测器
        self.detector_optim.zero_grad()
        outputs, _ = self.detector(signals)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.detector_optim.step()
        
        return f1, precision, recall, self.threshold
    
    def process_streaming_data(self, new_data, new_labels=None):
        # 转换为张量
        new_data = torch.FloatTensor(new_data).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)
        new_data = new_data.to(self.device)
        
        with torch.no_grad():
            outputs, features = self.detector(new_data)
            preds = (outputs > self.threshold).float().cpu().numpy()
        
        # 如果有新标签，进行在线学习
        if new_labels is not None:
            new_labels = torch.FloatTensor(new_labels).unsqueeze(0).to(self.device)
            f1, precision, recall, new_thresh = self.online_update(new_data.squeeze(0).cpu().numpy(), 
                                                                  new_labels.cpu().numpy())
            return preds, features, (f1, precision, recall, new_thresh)
        
        return preds, features, None