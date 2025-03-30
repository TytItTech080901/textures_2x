import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

class MinecraftTextureDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        
        # 递归获取所有PNG文件
        self.input_files = []
        self.target_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.png'):
                    rel_path = os.path.relpath(os.path.join(root, file), input_dir)
                    self.input_files.append(rel_path)
                    target_path = os.path.join(target_dir, rel_path)
                    if os.path.exists(target_path):
                        self.target_files.append(rel_path)
                    else:
                        raise FileNotFoundError(f"Target file {target_path} not found!")
        
        self.input_files.sort()
        self.target_files.sort()

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # 加载输入和目标图片
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        # 应用变换（如果有）
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        
        return input_img, target_img

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量，值范围[0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
])

# 示例：加载数据集
input_dir = 'dataset/input'  # 替换为实际路径
target_dir = 'dataset/target'  # 替换为实际路径
dataset = MinecraftTextureDataset(input_dir, target_dir, transform=transform)


# 数据集大小
dataset_size = len(dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)

# 划分比例
train_split = int(0.7 * dataset_size)
val_split = int(0.15 * dataset_size)
train_indices = indices[:train_split]
val_indices = indices[train_split:train_split + val_split]
test_indices = indices[train_split + val_split:]

# 创建数据加载器
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=16, sampler=test_sampler)

print(f"训练集大小: {len(train_indices)}")
print(f"验证集大小: {len(val_indices)}")
print(f"测试集大小: {len(test_indices)}")