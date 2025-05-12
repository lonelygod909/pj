import torch
import torch.multiprocessing as mp
import os
import signal
import gc

class SafeDataLoader:
    """安全的数据加载器包装类，处理多进程加载的问题"""
    
    @staticmethod
    def create_loader(dataset, batch_size, sampler=None, num_workers=2, 
                     collate_fn=None, pin_memory=True, **kwargs):
        """创建数据加载器，限制资源使用"""
        # 使用更保守的设置
        # 减少预加载的batch数量，默认值是2
        prefetch_factor = kwargs.pop('prefetch_factor', 1)
        
        # 确保工作进程正确退出
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=False,  # 不保持工作进程
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            **kwargs
        )
        return loader
    
    @staticmethod
    def close_loader(loader):
        """安全关闭数据加载器"""
        if loader is not None:
            # 确保关闭数据加载器的迭代器，如果存在的话
            if hasattr(loader, '_iterator'):
                try:
                    del loader._iterator
                except:
                    pass
            
            # 删除加载器对象
            del loader
            # 强制垃圾回收
            gc.collect()
            torch.cuda.empty_cache()

def setup_worker_sharing_strategy():
    """设置工作进程共享策略，减少内存使用"""
    # 对于Torch 1.9+，设置文件描述符共享策略
    if hasattr(mp, 'set_sharing_strategy'):
        mp.set_sharing_strategy('file_system')

def set_dataloader_environment():
    """设置与DataLoader相关的环境变量"""
    # 告诉pytorch使用更保守的工作进程设置
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
