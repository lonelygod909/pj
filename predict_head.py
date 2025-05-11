# input tensor: (1, 1000) which is the output of
# the encoder
# output tensor: (1) label
import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=1024):
        super(PredictHead, self).__init__()
        # 第一层: FC + BN + ELU
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.elu1 = nn.ELU(inplace=True)
        self.dropout1 = nn.Dropout(0.6)
        
        # 第二层: FC + BN + ELU + Dropout
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.elu2 = nn.ELU(inplace=True)
        self.dropout2 = nn.Dropout(0.6)
        
        # 第三层: FC (输出层)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.elu3 = nn.ELU(inplace=True)
        self.dropout3 = nn.Dropout(0.6)

        self.fc4 = nn.Linear(hidden_dim // 4, num_classes)

        # apply kaiming initialization
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        
        
        
    def forward(self, x):
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.dropout1(x)
        
        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.dropout2(x)
        
        # 第三层
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        # 输出层不使用 log_softmax
        return x