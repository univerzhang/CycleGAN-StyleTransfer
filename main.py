import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from mydatasets import CreateDatasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.tanh(self.conv5(x))
        return x

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv1(x), 0.2)
        x = nn.functional.leaky_relu(self.conv2(x), 0.2)
        x = nn.functional.leaky_relu(self.conv3(x), 0.2)
        x = nn.functional.sigmoid(self.conv4(x))
        return x


def main():

    # 训练集与验证集
    train = CreateDatasets(root_path='D:/BaiduNetdiskDownload/monet2photo', img_size=256, mode='train')
    val = CreateDatasets(root_path='D:/BaiduNetdiskDownload/monet2photo', img_size=256, mode='test')

    # 创建数据加载器
    train_dataloader = DataLoader(dataset=train, batch_size=1, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(dataset=val, batch_size=1, num_workers=4, drop_last=True)

    # 实例化模型
    generator_A = Generator().to(device)
    generator_B = Generator().to(device)
    discriminator_A = Discriminator().to(device)
    discriminator_B = Discriminator().to(device)

    # 定义优化器
    optimizer_G = optim.Adam([{'params': generator_A.parameters()}, {'params': generator_B.parameters()}])
    optimizer_D = optim.Adam([{'params': discriminator_A.parameters()}, {'params': discriminator_B.parameters()}])

    # 训练模型
    num_epochs = 1
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch}/{num_epochs}]')
        loop = tqdm(train_dataloader)  # 使用tqdm包装train_dataloader
        for i, data in enumerate(loop):
            inputs_A, inputs_B = data[0].to(device), data[1].to(device)

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            # 生成图像
            generated_B = generator_A(inputs_A)
            generated_A = generator_B(inputs_B)

            # 循环一致性损失
            cycled_A = generator_B(generated_B)
            cycled_B = generator_A(generated_A)

            # 判别器输出
            real_A_output = discriminator_A(inputs_A)
            real_B_output = discriminator_B(inputs_B)
            fake_A_output = discriminator_A(generated_A)
            fake_B_output = discriminator_B(generated_B)

            # 计算损失
            cycle_loss_A = torch.mean(torch.abs(inputs_A - cycled_A))
            cycle_loss_B = torch.mean(torch.abs(inputs_B - cycled_B))
            adversarial_loss_A = torch.mean(torch.abs(real_A_output - fake_A_output))
            adversarial_loss_B = torch.mean(torch.abs(real_B_output - fake_B_output))

            # 总损失
            total_loss_A = cycle_loss_A + adversarial_loss_A
            total_loss_B = cycle_loss_B + adversarial_loss_B

            # 反向传播和优化
            total_loss_A.backward(retain_graph=True)  # 设置retain_graph=True
            total_loss_B.backward()  # 第二次反向传播不需要设置retain_graph
            optimizer_G.step()
            optimizer_D.step()

            #loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(Loss_A=total_loss_A.item(), Loss_B=total_loss_B.item())

            # 验证循环
            with torch.no_grad():
                val_loop = tqdm(val_dataloader)  # 使用tqdm包装val_dataloader
                val_loss_A = 0
                val_loss_B = 0
                for j, val_data in enumerate(val_loop):
                    val_inputs_A, val_inputs_B = val_data[0].to(device), val_data[1].to(device)

                    # 生成图像
                    val_generated_B = generator_A(val_inputs_A)
                    val_generated_A = generator_B(val_inputs_B)

                    # 循环一致性损失
                    val_cycled_A = generator_B(val_generated_B)
                    val_cycled_B = generator_A(val_generated_A)

                    # 判别器输出
                    val_real_A_output = discriminator_A(val_inputs_A)
                    val_real_B_output = discriminator_B(val_inputs_B)
                    val_fake_A_output = discriminator_A(val_generated_A)
                    val_fake_B_output = discriminator_B(val_generated_B)

                    # 计算损失
                    val_cycle_loss_A = torch.mean(torch.abs(val_inputs_A - val_cycled_A))
                    val_cycle_loss_B = torch.mean(torch.abs(val_inputs_B - val_cycled_B))
                    val_adversarial_loss_A = torch.mean(torch.abs(val_real_A_output - val_fake_A_output))
                    val_adversarial_loss_B = torch.mean(torch.abs(val_real_B_output - val_fake_B_output))

                    # 总损失
                    val_total_loss_A = val_cycle_loss_A + val_adversarial_loss_A
                    val_total_loss_B = val_cycle_loss_B + val_adversarial_loss_B

                    val_loss_A += val_total_loss_A.item()
                    val_loss_B += val_total_loss_B.item()

                    val_loop.set_description(f'Validation Epoch [{epoch}/{num_epochs}]')
                    val_loop.set_postfix(Val_Loss_A=val_loss_A / (i + 1), Val_Loss_B=val_loss_B / (i + 1))

                print(f'Validation Loss A: {val_loss_A / (i + 1)}, Validation Loss B: {val_loss_B / (i + 1)}')


if __name__ == '__main__':
    main()