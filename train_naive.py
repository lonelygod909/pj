from dataset import *
from model import *
import pandas as pd
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
num_epochs = 4

image_dir = './train'
train_labels = "train_labels.csv"
train_labels = pd.read_csv(train_labels)
# to dict
train_labels = {row['id'] + '.tif': row['label'] for _, row in train_labels.iterrows()}
batch_size = 512
num_workers = 16

data_loader = DataLoader(image_dir, train_labels, batch_size, num_workers)
logger = TrainingLogger()

k_folds = 5
cv_loaders = data_loader.get_all_cv_loaders(k_folds=k_folds)

for fold, (train_loader, val_loader) in enumerate(cv_loaders):
    print(f"训练折 {fold+1}/{k_folds}")
    model = create_model(num_classes=1, model_name='hiera_base_224', freeze_encoder=True)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    # 训练模型
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 记录每个epoch的指标
        logger.log_epoch(epoch+1, train_loss, train_acc, val_loss, val_acc)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # 记录当前折的最终结果
    logger.log_fold(fold+1, train_loss, train_acc, val_loss, val_acc)

# 输出交叉验证的汇总结果
cv_summary = logger.summarize_cv_results()
print("交叉验证结果汇总:")
for key, value in cv_summary.items():
    print(f"{key}: {value:.4f}")

# 绘制训练过程图表
logger.plot_metrics()