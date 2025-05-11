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
from timeloader_utils import SafeDataLoader, setup_worker_sharing_strategy, set_dataloader_environment
import argparse
import gc
import signal
import gc
import time
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torchvision import transforms
import psutil
import sys
import socket
import datetime  # 正确导入整个datetime模块

def find_free_port():
    """查找一个可用的端口"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定端口0将让操作系统分配一个可用端口
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def setup(rank, world_size, port):
    """
    初始化分布式环境
    
    Args:
        rank: 当前进程的排名
        world_size: 总进程数
        port: 主进程的端口号
    """
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # 更详细的初始化过程报告
    print(f"Rank {rank}: 初始化进程组，PID={os.getpid()}, PORT={port}")
    try:
        # 增加超时时间，以确保有足够的时间建立连接
        dist.init_process_group(
            "nccl", 
            rank=rank, 
            world_size=world_size,
            timeout=datetime.timedelta(minutes=5),  # 增加到5分钟
            device_id=torch.device(f'cuda:{rank}')
        )
        
        # 明确设置当前进程使用的CUDA设备
        torch.cuda.set_device(rank)
        print(f"Rank {rank}: 进程组初始化成功，使用 GPU {rank}")
    except Exception as e:
        print(f"Rank {rank}: 进程组初始化失败 - {str(e)}")
        raise

def cleanup():
    """销毁分布式进程组"""
    if dist.is_initialized():
        print(f"PID {os.getpid()}: 销毁进程组")
        dist.destroy_process_group()
    else:
        print(f"PID {os.getpid()}: 进程组未初始化，跳过销毁")

def kill_child_processes(timeout=3):
    """杀死当前进程的所有子进程"""
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    
    if not children:
        return
    
    # 先尝试礼貌地结束进程
    for proc in children:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    
    # 等待子进程终止
    _, alive = psutil.wait_procs(children, timeout=timeout)
    
    # 如果仍有存活的子进程，强制结束它们
    for proc in alive:
        try:
            print(f"强制结束子进程: {proc.pid}")
            proc.kill()
        except psutil.NoSuchProcess:
            pass

def train_model(rank, world_size, port):
    # 记录进程ID和父进程ID
    pid = os.getpid()
    ppid = os.getppid() if hasattr(os, 'getppid') else None
    print(f"启动进程: Rank {rank}, PID={pid}, PPID={ppid}")
    
    # 忽略SIGTERM信号，确保进程不会被意外终止
    def handle_sigterm(signum, frame):
        print(f"Rank {rank} (PID {pid}): 收到信号 {signum}, 忽略...")
    
    # 捕获中断信号，确保资源正确释放
    def handle_sigint(signum, frame):
        print(f"Rank {rank} (PID {pid}): 收到中断信号 {signum}, 清理资源...")
        if dist.is_initialized():
            cleanup()
        kill_child_processes()
        sys.exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigint)
    
    try:
        # 设置工作进程共享策略
        if hasattr(mp, 'get_context'):
            mp.get_context('spawn')
        
        # 确保不同进程启动时间有错开，防止端口争用
        time.sleep(rank * 2.0)  # 增加更长的延迟
        
        # 设置随机种子
        seed = 4 + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        # 降低预缓存并清理CUDA缓存
        torch.cuda.empty_cache()
        
        # 设置分布式环境
        setup(rank, world_size, port)
        device = torch.device(f"cuda:{rank}")
                
        
        batch_size = 64  # 更小的批次大小
        num_epochs = 16  # 增加训练轮数
        image_dir = '../../autodl-tmp/data/train'
        train_labels = "train_labels.csv"
        train_labels = pd.read_csv(train_labels)
        train_labels = {row['id'] + '.tif': row['label'] for _, row in train_labels.iterrows()}
        num_workers = 8   # 减少工作线程，避免资源争用
        
        k_folds = 5

        # 创建数据集和采样器
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomVerticalFlip(), 
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 高斯模糊           
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])

        # 验证集转换(只有基本操作，没有随机增强)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 为训练和验证创建不同的数据集
        train_dataset = TumorDataset(image_dir, train_labels, transform=train_transform)
        val_dataset = TumorDataset(image_dir, train_labels, transform=val_transform)
            
        # 创建K折交叉验证
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        indices = list(range(len(train_dataset)))
        fold_splits = list(kfold.split(indices))
        
        if rank == 0:
            logger = TrainingLogger(log_dir="logs_2")
        
        # 确保所有进程都完成日志记录器的初始化
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                print(f"Rank {rank}: 同步屏障超时 - {str(e)}")
                # 尝试重新初始化
                if dist.is_initialized():
                    cleanup()
                time.sleep(5)
                setup(rank, world_size, port)
        else:
            print(f"Rank {rank}: 进程组未初始化，跳过同步屏障")
            return

        # 记录每个折的结果
        fold_results = []
        
        # 进行K折训练
        for fold in range(k_folds):
            # 每个折前先验证进程状态
            if not dist.is_initialized():
                print(f"Rank {rank}: 进程组未初始化，尝试重新初始化...")
                try:
                    setup(rank, world_size, port)
                    time.sleep(2)  # 等待初始化稳定
                except Exception as e:
                    print(f"Rank {rank}: 重新初始化失败 - {str(e)}")
                    break
            
            # 验证进程状态
            try:
                actual_rank = dist.get_rank()
                if actual_rank != rank:
                    print(f"警告: 进程 PID={pid} 的 rank 不一致: 预期={rank}, 实际={actual_rank}")
                    continue
            except Exception as e:
                print(f"Rank {rank}: 获取实际rank失败 - {str(e)}")
                # 尝试重新初始化
                try:
                    if dist.is_initialized():
                        cleanup()
                    time.sleep(5)
                    setup(rank, world_size, port)
                    time.sleep(2)
                except:
                    break
                
            if rank == 0:
                print(f"\n开始训练折 {fold+1}/{k_folds}")
                
            train_indices, val_indices = fold_splits[fold]
            
            # 创建训练子集
            train_subset = torch.utils.data.Subset(train_dataset, train_indices)
            
            # 创建分布式采样器
            train_sampler = DistributedSampler(
                train_subset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False  # 不丢弃最后一批数据
            )
            
            # 创建训练数据加载器，禁用persistent_workers以防止内存泄漏
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                collate_fn=CustomCollate(),
                persistent_workers=False,  # 确保每个折后工作进程被关闭
                pin_memory=False,  # 关闭pin_memory减少内存使用
                prefetch_factor=1 if num_workers > 0 else None,  # 减少预取数量
            )
            
            # 验证集不需要分布式，只在主进程创建
            val_loader = None
            if rank == 0:
                val_subset = torch.utils.data.Subset(val_dataset, val_indices)
                val_loader = torch.utils.data.DataLoader(
                    val_subset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=CustomCollate(),
                    persistent_workers=False,  # 确保每个折后工作进程被关闭
                    pin_memory=False  # 关闭pin_memory减少内存使用
                )
                print(f"训练集大小: {len(train_subset)}, 验证集大小: {len(val_subset)}")
            
            # 同步点，确保所有进程都准备好数据
            try:
                torch.cuda.synchronize(device)  # 同步GPU
                dist.barrier()
            except Exception as e:
                print(f"Rank {rank}: 同步屏障失败 - {str(e)}")
                # 尝试恢复分布式环境
                try:
                    if dist.is_initialized():
                        cleanup()
                    time.sleep(5)
                    setup(rank, world_size, port)
                except:
                    break
            
            # 创建模型 - 使用较小的模型减少内存
            model = create_model(num_classes=1, model_name='hiera_large_224', freeze_encoder=False)  # 使用更小的模型
            model = model.to(device)
            
            # 使用DDP包装模型，禁用广播缓冲区以减少通信
            model = DDP(
                model, 
                device_ids=[rank], 
                find_unused_parameters=False,
                broadcast_buffers=False,  # 禁用缓冲区广播以减少通信
                gradient_as_bucket_view=True  # 减少内存使用
            )
            
            # 创建损失函数
            criterion = nn.BCEWithLogitsLoss().to(device)
            
            # 修改优化器配置
            optimizer = torch.optim.AdamW([
                {'params': model.module.encoder.parameters(), 'lr': 1e-4},
                {'params': model.module.predict_head.parameters(), 'lr': 5e-4}
            ], weight_decay=0.001)
            
            # 使用简单的学习率调度器
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[1e-3, 5e-3],  # 对应 optimizer 中不同参数组的最大 LR
                total_steps=len(train_loader) * num_epochs,  # 总训练步数（必须精确）
                pct_start=0.1,        # 前10%的训练步骤用于学习率从初始值升到最大值（预热阶段）
                div_factor=25,        # 初始学习率 = max_lr / 25
                final_div_factor=1000 # 最终学习率 = max_lr / (25 * 1000)
            )

            
            if rank == 0:
                print(f"Fold {fold+1} - 模型和优化器初始化完成")
            
            # 记录每个epoch的指标
            best_val_acc = 0
            
            # 训练模型
            for epoch in range(num_epochs):
                # 检查进程状态
                if not dist.is_initialized():
                    print(f"警告: Rank {rank} (PID {pid}) 的进程组已被销毁，重新初始化")
                    try:
                        setup(rank, world_size, port)
                    except Exception as e:
                        print(f"重新初始化失败: {str(e)}")
                        break
                
                # 后面的两个epoch解冻encoder
                if epoch >= 8:
                    if rank == 0:
                        print("解冻编码器...")
                    model.module.unfreeze_encoder()
                
                # 设置epoch，确保每个epoch数据洗牌不同
                train_sampler.set_epoch(epoch)
                
                # 训练一个epoch
                try:
                    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, rank, device, scheduler=scheduler)
                    
                    # 每次epoch后清理缓存
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Rank {rank} (PID {pid}) 训练epoch时发生错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    break
                
                # 只在主进程(rank 0)进行评估和日志记录
                val_loss = 0
                val_acc = 0
                if rank == 0:
                    try:
                        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                        logger.log_epoch(epoch+1, train_loss, train_acc, val_loss, val_acc)
                        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                        
                        # 保存最佳模型
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            torch.save(model.module.state_dict(), f"model_fold_{fold+1}.pth")
                            print(f"保存模型: Fold {fold+1}, Epoch {epoch+1}, Val Acc: {val_acc:.2f}%")
                    except Exception as e:
                        print(f"Rank {rank} (PID {pid}) 评估时发生错误: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # 同步所有进程，确保一个epoch完成
                torch.cuda.synchronize(device)
                dist.barrier()
            
            # 当前epoch完成后立即关闭数据加载器
            print(f"Rank {rank} (PID {pid}) 关闭数据加载器")
            # 使用完后先将它们设为None，以便手动触发垃圾回收
            if hasattr(train_loader, '_iterator'):
                del train_loader._iterator
            del train_loader
                
            if rank == 0 and val_loader is not None:
                if hasattr(val_loader, '_iterator'):
                    del val_loader._iterator
                del val_loader
                
            # 当前折训练结束，记录结果
            if rank == 0:
                fold_results.append({
                    'fold': fold+1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc
                })
                logger.log_fold(fold+1, train_loss, train_acc, val_loss, val_acc)
            
            # 清理资源前同步
            try:
                torch.cuda.synchronize(device)
                dist.barrier()
            except Exception as e:
                print(f"Rank {rank}: 清理前同步屏障失败 - {str(e)}")
            
            # 清理资源，确保每个进程都完整地清理
            if 'model' in locals():
                model.to('cpu')
                del model
            
            if 'optimizer' in locals():
                del optimizer
            
            if 'scheduler' in locals():
                del scheduler
            
            if 'train_sampler' in locals():
                del train_sampler
            
            # 强制垃圾回收
            gc.collect()
            torch.cuda.empty_cache()
            
            # 杀死所有子进程
            kill_child_processes()
            
            # 等待资源释放
            time.sleep(5)
            
            # 检查当前进程占用的内存
            process = psutil.Process(os.getpid())
            print(f"Rank {rank} (PID {pid}) 内存占用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            # 确保所有进程完成资源清理
            try:
                torch.cuda.synchronize(device)
                dist.barrier()
            except Exception as e:
                print(f"Rank {rank}: 资源清理后同步屏障失败 - {str(e)}")
                break
            
            if rank == 0:
                print(f"完成第 {fold+1} 折训练，内存已清理")
        
        # 输出交叉验证的汇总结果
        if rank == 0:
            try:
                cv_summary = logger.summarize_cv_results()
                print("\n交叉验证结果汇总:")
                for key, value in cv_summary.items():
                    print(f"{key}: {value:.4f}")
                
                # 绘制训练过程图表
                logger.plot_metrics()
                
                # 找出最佳的折
                best_fold = max(fold_results, key=lambda x: x['best_val_acc'])
                print(f"\n最佳模型来自第 {best_fold['fold']} 折，验证准确率: {best_fold['best_val_acc']:.4f}")
                
                # 复制最佳模型为最终模型
                os.system(f"cp model_fold_{best_fold['fold']}.pth final_model_s.pth")
                print("已保存最终模型: final_model_s.pth")
            except Exception as e:
                print(f"汇总结果时出错: {str(e)}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"Rank {rank} (PID {pid}) 遇到错误: {e}")
        import traceback
        traceback.print_exc()
        if rank == 0 and 'logger' in locals():
            logger.log_error(e)
    
    finally:
        print(f"Rank {rank} (PID {pid}) 清理资源...")
        # 清理所有剩余资源
        for name in list(locals()):
            if name not in ['rank', 'dist', 'cleanup', 'port', 'world_size']:
                try:
                    del locals()[name]
                except:
                    pass
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # 杀死所有子进程
        kill_child_processes()
        
        if dist.is_initialized():
            cleanup()
        
        print(f"Rank {rank} (PID {pid}) 完成清理")

if __name__ == "__main__":
    
    # 不使用fork方法，避免进程ID冲突
    if hasattr(mp, 'set_start_method'):
        mp.set_start_method('spawn', force=True)
    
    # 获取可用的GPU数量，但限制使用2个GPU以减少资源争用
    world_size = min(2, torch.cuda.device_count())
    
    # 查找一个可用的空闲端口
    port = find_free_port()
    print(f"使用端口 {port} 进行分布式训练")
    
    # 确保CUDA可用且GPU数量足够
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        sys.exit(1)
        
    if torch.cuda.device_count() < world_size:
        print(f"警告: 需要 {world_size} 个GPU，但只找到 {torch.cuda.device_count()} 个")
        world_size = torch.cuda.device_count()
    
    print(f"使用 {world_size} 个GPU进行训练，GPU列表:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        # 显示每个GPU的显存
        print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")
    
    # 使用启动方法确保每个进程有唯一的进程ID
    try:
        # 清理环境变量，避免继承旧的设置
        if 'MASTER_ADDR' in os.environ:
            del os.environ['MASTER_ADDR']
        if 'MASTER_PORT' in os.environ:
            del os.environ['MASTER_PORT']
        
        # 预先清理CUDA缓存    
        torch.cuda.empty_cache()
            
        mp.spawn(
            train_model,
            args=(world_size, port),
            nprocs=world_size,
            join=True
        )
    except KeyboardInterrupt:
        print("接收到用户中断，终止训练...")
        # 强制终止所有相关进程
        parent = psutil.Process(os.getpid())
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except:
                pass
