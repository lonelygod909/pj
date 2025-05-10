# input tensor: (1, 1000) which is the output of
# the encoder
# output tensor: (1) label
import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super(PredictHead, self).__init__()
        # 第一层: FC + BN + ELU
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.elu1 = nn.ELU(inplace=True)
        
        # 第二层: FC + BN + ELU + Dropout
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.elu2 = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        # 第三层: FC (输出层)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        
        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.dropout(x)
        
        # 第三层
        x = self.fc3(x)
        
        # 输出层不使用 log_softmax
        return x