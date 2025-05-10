import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dataset import *
from model import *
import pandas as pd


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model(rank, world_size):
    try:
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # 设置分布式环境
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        
        # 配置
        num_epochs = 4
        image_dir = './train'
        train_labels = "train_labels.csv"
        train_labels = pd.read_csv(train_labels)
        train_labels = {row['id'] + '.tif': row['label'] for _, row in train_labels.iterrows()}
        batch_size = 32  # 每个GPU的批次大小
        num_workers = 2   # 每个GPU的工作线程
        
        k_folds = 5
        
        # 创建数据集和采样器
        dataset = TumorDataset(image_dir, train_labels, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]))
        
        # 创建K折交叉验证
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        indices = list(range(len(dataset)))
        fold_splits = list(kfold.split(indices))
        
        if rank == 0:
            logger = TrainingLogger()
        
        # 进行K折训练
        for fold in range(k_folds):
            train_indices, val_indices = fold_splits[fold]
            
            # 创建分布式采样器
            train_sampler = DistributedSampler(
                SubsetRandomSampler(train_indices),
                num_replicas=world_size,
                rank=rank
            )
            
            if 'train_loader' in locals():
                del train_loader
            

            train_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                collate_fn=CustomCollate(),
                persistent_workers=False
            )
            
            # 验证集不需要分布式
            if rank == 0:
                if 'val_loader' in locals():  
                    del val_loader
                val_loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    sampler=SubsetRandomSampler(val_indices),
                    num_workers=num_workers,
                    collate_fn=CustomCollate()
                )
                
                print(f"训练折 {fold+1}/{k_folds}")
            
            if 'model' in locals():
                del model
                
            torch.cuda.empty_cache()  # 清空CUDA缓存
            # 创建模型
            model = create_model(num_classes=1, model_name='hiera_base_224', freeze_encoder=True)
            model = model.to(device)
            model = DDP(model, device_ids=[rank])
            
            criterion = nn.BCEWithLogitsLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
            # 训练模型
            for epoch in range(num_epochs):
                # 设置epoch，确保每个epoch数据洗牌不同
                train_sampler.set_epoch(epoch)
                
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, rank, device, scheduler=scheduler)
                
                # 只在主进程(rank 0)进行评估和日志记录
                if rank == 0:
                    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                    logger.log_epoch(epoch+1, train_loss, train_acc, val_loss, val_acc)
                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 记录当前折的最终结果
            if rank == 0:
                logger.log_fold(fold+1, train_loss, train_acc, val_loss, val_acc)
        
        # 输出交叉验证的汇总结果
        if rank == 0:
            cv_summary = logger.summarize_cv_results()
            print("交叉验证结果汇总:")
            for key, value in cv_summary.items():
                print(f"{key}: {value:.4f}")
            
            # 绘制训练过程图表
            logger.plot_metrics()

    except Exception as e:
        print(f"Rank {rank} encountered an error: {e}")
        if rank == 0:
            logger.log_error(e)
        if dist.is_initialized():
            cleanup()   
    finally:
        if dist.is_initialized():
            cleanup()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    mp.set_start_method('spawn', force=True)
    # 获取可用的GPU数量
    world_size = 4
    print(f"使用 {world_size} 个GPU进行训练")
    
    # 启动多进程训练
    mp.spawn(
        train_model,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )