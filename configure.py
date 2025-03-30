import torch
import torch.nn as nn

class RRDB(nn.Module):
    def __init__(self, num_features=64, num_blocks=3):
        super(RRDB, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers += [nn.Conv2d(num_features, num_features, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
        self.rrdb = nn.Sequential(*layers)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        x = self.rrdb(x)
        return self.lrelu(x * 0.2 + residual)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=23):
        super(Generator, self).__init__()
        # 初始卷积层
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        # RRDB 主干
        trunk = []
        for _ in range(num_blocks):
            trunk.append(RRDB(num_features))
        self.trunk = nn.Sequential(*trunk)
        
        # 上采样层（调整为 2x）
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),  # 2x 上采样
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 输出层
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.trunk(feat)
        feat = feat + trunk
        feat = self.upsample(feat)
        out = self.conv_last(feat)
        return out
    

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        layers = [
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out
    

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 加载预训练权重
pretrained_dict = torch.load('models/RRDB_ESRGAN_x4.pth', map_location=device)

# 过滤掉不需要的层（因为我们调整了上采样）
model_dict = generator.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
generator.load_state_dict(model_dict)

print("预训练模型加载完成！")


# 测试输入
input_tensor = torch.randn(1, 3, 16, 16).to(device)
output_tensor = generator(input_tensor)
print(f"输入形状: {input_tensor.shape}")
print(f"输出形状: {output_tensor.shape}")  # 应为 [1, 3, 32, 32]