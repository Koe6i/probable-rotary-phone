import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
class SlidingWindowDataset(Dataset):
    def __init__(self, signal, window_size=64, step=16, labels=None):
        self.signal = signal
        self.window_size = window_size
        self.step = step
        self.labels = labels
        self.has_labels = labels is not None
        
    def __len__(self):
        return (len(self.signal) - self.window_size) // self.step + 1
    
    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.window_size
        segment = self.signal[start:end]
        
        if self.has_labels:
            # 使用窗口中间点的标签
            label = self.labels[start + self.window_size // 2]
            return torch.FloatTensor(segment).unsqueeze(0), torch.FloatTensor([label])
        return torch.FloatTensor(segment).unsqueeze(0)

def create_data_loaders(data, labels, batch_size=32, test_size=0.2):
    # 分割训练集和测试集
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, shuffle=False)
    
    train_dataset = SlidingWindowDataset(train_data, labels=train_labels)
    test_dataset = SlidingWindowDataset(test_data, labels=test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader