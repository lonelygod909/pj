# create dataloader
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
from predict_head import PredictHead
import hiera
from tqdm import tqdm
from sklearn.model_selection import KFold
import json
import matplotlib.pyplot as plt
from datetime import datetime

class TumorDataset(Dataset):
    def __init__(self, image_dir, train_labels, transform=None):
        """
        Input:
            image_dir: Directory containing images, in this case images are tif files.
            train_labels: Labels for the images, dictionary with image names as keys and labels as values.
            transform: Transformations to be applied to the images.
        """
        self.image_dir = image_dir
        self.train_labels = train_labels
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        # image is tif
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取文件名并从字典中查找标签
        filename = self.image_filenames[idx]
        # 检查文件名是否在标签字典中
        if filename not in self.train_labels:
            raise KeyError(f"找不到文件 '{filename}' 的标签。可用的键: {list(self.train_labels.keys())[:5]}...")
        
        label = self.train_labels[filename]
        
        if self.transform:
            # 将 numpy 数组转换为 PIL 图像
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label

class CustomCollate:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, batch):
        # labels in as a one dimension numpy array
        images, labels = zip(*batch)
                
        
        return torch.stack(images), torch.tensor(labels).unsqueeze(1)

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{self.timestamp}.json")
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "fold_results": []
        }
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
        """记录每个epoch的训练和验证指标"""
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc
        }
        
        if val_loss is not None:
            epoch_log["val_loss"] = val_loss
        if val_acc is not None:
            epoch_log["val_acc"] = val_acc
            
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        
        if val_loss is not None:
            self.history["val_loss"].append(val_loss)
        if val_acc is not None:
            self.history["val_acc"].append(val_acc)
        
        self._save_log()
        return epoch_log
    
    def log_fold(self, fold, final_train_loss, final_train_acc, final_val_loss, final_val_acc):
        """记录每个交叉验证折的最终结果"""
        fold_result = {
            "fold": fold,
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc
        }
        self.history["fold_results"].append(fold_result)
        self._save_log()
        return fold_result
    
    def _save_log(self):
        """保存日志到文件"""
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def plot_metrics(self, save=True):
        """绘制训练和验证指标曲线"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        if len(self.history["val_loss"]) > 0:
            plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label="Train Accuracy")
        if len(self.history["val_acc"]) > 0:
            plt.plot(self.history["val_acc"], label="Validation Accuracy")
        plt.legend()
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        
        plt.tight_layout()
        
        if save:
            plot_file = os.path.join(self.log_dir, f"metrics_plot_{self.timestamp}.png")
            plt.savefig(plot_file)
            
        plt.show()
        
    def summarize_cv_results(self):
        """汇总交叉验证结果"""
        if len(self.history["fold_results"]) == 0:
            return "No cross-validation results available."
        
        train_losses = [fold["final_train_loss"] for fold in self.history["fold_results"]]
        train_accs = [fold["final_train_acc"] for fold in self.history["fold_results"]]
        val_losses = [fold["final_val_loss"] for fold in self.history["fold_results"]]
        val_accs = [fold["final_val_acc"] for fold in self.history["fold_results"]]
        
        summary = {
            "mean_train_loss": np.mean(train_losses),
            "std_train_loss": np.std(train_losses),
            "mean_train_acc": np.mean(train_accs),
            "std_train_acc": np.std(train_accs),
            "mean_val_loss": np.mean(val_losses),
            "std_val_loss": np.std(val_losses),
            "mean_val_acc": np.mean(val_accs),
            "std_val_acc": np.std(val_accs)
        }
        
        self.history["cv_summary"] = summary
        self._save_log()
        
        return summary

class DataLoader:
    def __init__(self, image_dir, train_labels, batch_size=32, num_workers=4):
        self.image_dir = image_dir
        self.train_labels = train_labels   
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def get_loader(self, indices=None, shuffle=True):
        """
        获取数据加载器，可以指定样本索引以支持交叉验证
        
        Args:
            indices: 要包含的样本索引，如果为None则使用全部样本
            shuffle: 是否打乱数据
        """
        dataset = TumorDataset(self.image_dir, self.train_labels, transform=self.transform)
        
        if indices is not None:
            sampler = SubsetRandomSampler(indices)
            return torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers, 
                collate_fn=CustomCollate(self.transform)
            )
        else:
            return torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=shuffle,
                num_workers=self.num_workers, 
                collate_fn=CustomCollate(self.transform)
            )
    
    def get_cv_loaders(self, k_folds=5, val_fold=0):
        """
        获取用于交叉验证的训练和验证数据加载器
        
        Args:
            k_folds: 交叉验证的折数
            val_fold: 当前用作验证集的折索引 (0 to k_folds-1)
            
        Returns:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        dataset = TumorDataset(self.image_dir, self.train_labels, transform=self.transform)
        
        # 创建交叉验证分割器
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # 获取所有索引
        indices = list(range(len(dataset)))
        
        # 获取当前折的训练和验证索引
        fold_splits = list(kfold.split(indices))
        train_indices, val_indices = fold_splits[val_fold]
        
        # 创建训练和验证数据加载器
        train_loader = self.get_loader(indices=train_indices)
        val_loader = self.get_loader(indices=val_indices)
        
        return train_loader, val_loader
    
    def get_all_cv_loaders(self, k_folds=5):
        """
        获取所有交叉验证折的加载器
        
        Args:
            k_folds: 交叉验证的折数
            
        Returns:
            cv_loaders: 包含每一折训练和验证加载器的列表
        """
        cv_loaders = []
        
        for fold in range(k_folds):
            train_loader, val_loader = self.get_cv_loaders(k_folds=k_folds, val_fold=fold)
            cv_loaders.append((train_loader, val_loader))
        
        return cv_loaders


def train_one_epoch(model, dataloader, criterion, optimizer, rank, device, scheduler=None):
    """
    训练模型一个 epoch
    
    Args:
        model: 要训练的模型
        dataloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 运行设备 (cuda/cpu)
        scheduler: 学习率调度器 (可选)
    
    Returns:
        epoch_loss: 本 epoch 的平均损失
        epoch_acc: 本 epoch 的准确率 (%)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm创建进度条
    pbar = tqdm(dataloader, desc=f"训练中rank{rank}")
    for i, (images, labels) in enumerate(pbar):
        try:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播 - 直接通过模型，不需要分离编码器和预测头
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            loss = criterion(outputs, labels.float())
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            # 统计损失和准确率
            running_loss += loss.item()

            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()
            
            # 更新进度条
            current_loss = running_loss / (i + 1)
            current_acc = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
        
        except Exception as e:
            print(f"训练中发生错误: {str(e)}")
            continue
    
    # 计算平均损失和准确率
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """
    在验证/测试集上评估模型
    
    Args:
        model: 要评估的模型
        dataloader: 验证/测试数据加载器
        criterion: 损失函数
        device: 运行设备 (cuda/cpu)
    
    Returns:
        epoch_loss: 平均损失
        epoch_acc: 准确率 (%)
        predictions: 预测结果 (可选，如需返回所有预测结果)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="评估中"):
            try:
                images = images.to(device)
                labels = labels.to(device)
                
                # 前向传播 - 直接通过模型，不需要分离编码器和预测头
                outputs = model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                loss = criterion(outputs, labels.float())
            
                # 统计损失和准确率
                running_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels.float()).sum().item()  # 修正为 labels.float()
                
                # 保存预测结果和标签（可用于后续分析）
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            except Exception as e:
                print(f"评估中发生错误: {str(e)}")
                continue
    
    # 计算平均损失和准确率
    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    epoch_acc = 100 * correct / total if total > 0 else 0
    
    return epoch_loss, epoch_acc

