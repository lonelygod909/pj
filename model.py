import torch
import torch.nn as nn
from predict_head import PredictHead
import hiera


class HieraClassifier(nn.Module):
    """
    使用Hiera模型作为特征提取器（编码器）和PredictHead作为分类头（解码器）的分类器模型
    """
    def __init__(self, num_classes, feature_dim=1000, hidden_dim=512, model_name='hiera_base_224', pretrained=True):
        """
        初始化HieraClassifier模型
        
        Args:
            num_classes: 分类类别数
            feature_dim: Hiera模型输出特征维度，默认为1000
            hidden_dim: PredictHead中隐藏层的维度，默认为512
            model_name: 使用的Hiera模型名称，默认为'hiera_base_224'
            pretrained: 是否使用预训练的Hiera模型，默认为True
        """
        super(HieraClassifier, self).__init__()
        
        # 初始化编码器（Hiera模型）
        if model_name == 'hiera_base_224':
            self.encoder = hiera.hiera_base_224(pretrained=pretrained)
        elif model_name == 'hiera_large_224':
            self.encoder = hiera.hiera_large_224(pretrained=pretrained)
        elif model_name == 'hiera_huge_224':
            self.encoder = hiera.hiera_huge_224(pretrained=pretrained)
        else:
            raise ValueError(f"不支持的Hiera模型: {model_name}")
        
        # 初始化解码器（PredictHead）
        self.predict_head = PredictHead(feature_dim, num_classes, hidden_dim=hidden_dim)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量，形状为 [batch_size, channels, height, width]
        
        Returns:
            模型输出，形状为 [batch_size, num_classes]
        """
        # 获取特征表示
        features = self.encoder(x)
        
        # 使用预测头进行分类， 输出是概率，要转化成标签
        outputs = self.predict_head(features) 

        # print(predicted_labels.shape)
        return outputs  # 返回形状为 [batch_size, 1] 的标签
    
    def freeze_encoder(self):
        """冻结编码器参数，只训练解码器（预测头）"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """解冻编码器参数，允许训练整个模型"""
        for param in self.encoder.parameters():
            param.requires_grad = True
            
    def get_trainable_parameters(self):
        """获取所有可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]


# 辅助函数，用于在train_naive.py中方便地创建模型
def create_model(num_classes, model_name='hiera_base_224', freeze_encoder=True):
    """
    创建HieraClassifier模型的辅助函数
    
    Args:
        num_classes: 分类类别数
        model_name: 使用的Hiera模型名称
        freeze_encoder: 是否冻结编码器参数
        
    Returns:
        初始化的HieraClassifier模型
    """
    model = HieraClassifier(num_classes=num_classes, model_name=model_name)
    
    if freeze_encoder:
        model.freeze_encoder()
    
    return model