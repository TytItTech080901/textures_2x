import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from configure import device,generator,discriminator
from dataset import train_loader,val_loader


# 对抗损失
criterion_gan = nn.BCEWithLogitsLoss()

# 内容损失
criterion_content = nn.L1Loss()

# 感知损失（可选，使用 VGG19）
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

def perceptual_loss(fake_img, real_img, vgg):
    layers = {'3': 0.1, '8': 0.3, '15': 0.6}  # VGG19 的第 3、8、15 层
    loss = 0
    for name, module in vgg.named_children():
        fake_img = module(fake_img)
        real_img = module(real_img)
        if name in layers:
            loss += criterion_content(fake_img, real_img) * layers[name]
        if name == '15':
            break
    return loss

# 优化器
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))



# 训练参数
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练循环
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        lr_imgs = lr_imgs.to(device)  # 16x16 输入
        hr_imgs = hr_imgs.to(device)  # 32x32 目标
        
        # ----- 训练判别器 -----
        optimizer_D.zero_grad()
        
        # 真实图像的判别
        real_output = discriminator(hr_imgs)
        real_labels = torch.ones_like(real_output).to(device)
        d_loss_real = criterion_gan(real_output, real_labels)
        
        # 生成图像的判别
        fake_imgs = generator(lr_imgs)
        fake_output = discriminator(fake_imgs.detach())  # detach 避免影响生成器
        fake_labels = torch.zeros_like(fake_output).to(device)
        d_loss_fake = criterion_gan(fake_output, fake_labels)
        
        # 总判别器损失
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # ----- 训练生成器 -----
        optimizer_G.zero_grad()
        
        # 对抗损失
        fake_output = discriminator(fake_imgs)
        g_loss_gan = criterion_gan(fake_output, real_labels)
        
        # 内容损失
        g_loss_content = criterion_content(fake_imgs, hr_imgs)
        
        # 感知损失（可选）
        g_loss_perceptual = perceptual_loss(fake_imgs, hr_imgs, vgg) if 'vgg' in globals() else 0
        
        # 总生成器损失（调整权重）
        g_loss = g_loss_gan + 100.0 * g_loss_content + 10.0 * g_loss_perceptual
        g_loss.backward()
        optimizer_G.step()
        
        # 打印训练进度
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(train_loader)}] "
                  f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")
    
    # 验证（可选）
    generator.eval()
    val_loss = 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            fake_imgs = generator(lr_imgs)
            val_loss += criterion_content(fake_imgs, hr_imgs).item()
    val_loss /= len(val_loader)
    print(f"Epoch [{epoch}/{num_epochs}] Validation Loss: {val_loss:.4f}")
    
    # 保存模型
    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), f'models/generator_epoch_{epoch+1}.pth')
        torch.save(discriminator.state_dict(), f'models/discriminator_epoch_{epoch+1}.pth')