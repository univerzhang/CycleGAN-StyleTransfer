import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
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

    train = CreateDatasets(root_path='D:/BaiduNetdiskDownload/monet2photo/trainA', img_size=256, mode='train')
    # 创建数据加载器
    dataloader = DataLoader(dataset=train, batch_size=1, shuffle=False, num_workers=4,
                            drop_last=True)
    print(len(dataloader))

    # 实例化模型
    generator_A = Generator().to(device)
    generator_B = Generator().to(device)
    discriminator_A = Discriminator().to(device)
    discriminator_B = Discriminator().to(device)

    # 定义优化器
    optimizer_G = optim.Adam([{'params': generator_A.parameters()}, {'params': generator_B.parameters()}])
    optimizer_D = optim.Adam([{'params': discriminator_A.parameters()}, {'params': discriminator_B.parameters()}])

    # 训练模型
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch}/{num_epochs}]')
        loop = tqdm(dataloader)  # 使用tqdm包装dataloader
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

            print(f'Loss A: {total_loss_A.item()}, Loss B: {total_loss_B.item()}')
        print(f'Epoch [{epoch}/{num_epochs}]')


if __name__ == '__main__':
    main()