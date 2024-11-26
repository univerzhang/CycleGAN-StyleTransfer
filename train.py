import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from mydatasets import CreateDatasets
from model import Generator, Discriminator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
datasets_path = 'D:/BaiduNetdiskDownload/monet2photo'


def train(epoch, num_epochs, train_dataloader, O_G, O_D, G_A, G_B, D_A, D_B):
    print(f'Epoch [{epoch}/{num_epochs}]')
    loop = tqdm(train_dataloader)  # 使用tqdm包装train_dataloader
    for _, data in enumerate(loop):
        inputs_A, inputs_B = data[0].to(device), data[1].to(device)

        O_G.zero_grad()
        O_D.zero_grad()

        # 生成图像
        generated_B = G_A(inputs_A)
        generated_A = G_B(inputs_B)

        # 循环一致性损失
        cycled_A = G_B(generated_B)
        cycled_B = G_A(generated_A)

        # 判别器输出
        real_A_output = D_A(inputs_A)
        real_B_output = D_B(inputs_B)
        fake_A_output = D_A(generated_A)
        fake_B_output = D_B(generated_B)

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
        O_G.step()
        O_D.step()

        # loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        loop.set_postfix(Loss_A=total_loss_A.item(), Loss_B=total_loss_B.item())

def verify(epoch, num_epochs, val_dataloader, G_A, G_B, D_A, D_B):
    val_loop = tqdm(val_dataloader)  # 使用tqdm包装val_dataloader
    val_loss_A = 0
    val_loss_B = 0
    for _, val_data in enumerate(val_loop):
        val_inputs_A, val_inputs_B = val_data[0].to(device), val_data[1].to(device)

        # 生成图像
        val_generated_B = G_A(val_inputs_A)
        val_generated_A = G_B(val_inputs_B)

        # 循环一致性损失
        val_cycled_A = G_B(val_generated_B)
        val_cycled_B = G_A(val_generated_A)

        # 判别器输出
        val_real_A_output = D_A(val_inputs_A)
        val_real_B_output = D_B(val_inputs_B)
        val_fake_A_output = D_A(val_generated_A)
        val_fake_B_output = D_B(val_generated_B)

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
        # val_loop.set_postfix(Val_Loss_A=val_loss_A / (i + 1), Val_Loss_B=val_loss_B / (i + 1))

    # print(f'Validation Loss A: {val_loss_A / (i + 1)}, Validation Loss B: {val_loss_B / (i + 1)}')


def save(num_epochs, O_G, O_D, G_A, G_B, D_A, D_B):
    torch.save({
        'generator_A': G_A.state_dict(),
        'generator_B': G_B.state_dict(),
        'discriminator_A': D_A.state_dict(),
        'discriminator_B': D_B.state_dict(),
        'optimizer_G': O_G.state_dict(),
        'optimizer_D': O_D.state_dict(),
        'epoch': num_epochs
    }, './cycleGAN_model.pth')


def main():

    # 训练集与验证集
    train_set = CreateDatasets(root_path=datasets_path, img_size=256, mode='train')
    val_set = CreateDatasets(root_path=datasets_path, img_size=256, mode='test')

    # 创建数据加载器
    train_dataloader = DataLoader(dataset=train_set, batch_size=1, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(dataset=val_set, batch_size=1, num_workers=4, drop_last=True)

    # 实例化模型
    generator_A = Generator().to(device)
    generator_B = Generator().to(device)
    discriminator_A = Discriminator().to(device)
    discriminator_B = Discriminator().to(device)

    # 定义优化器
    optimizer_G = optim.Adam([{'params': generator_A.parameters()}, {'params': generator_B.parameters()}])
    optimizer_D = optim.Adam([{'params': discriminator_A.parameters()}, {'params': discriminator_B.parameters()}])

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        train(epoch, num_epochs, train_dataloader, optimizer_G, optimizer_D,
              generator_A, generator_B, discriminator_A, discriminator_B)

        # 验证循环
        with torch.no_grad():
            verify(epoch, num_epochs, val_dataloader,
                   generator_A, generator_B, discriminator_A,discriminator_B)

        # 保存模型
        save(num_epochs, optimizer_G, optimizer_D,
             generator_A, generator_B, discriminator_A, discriminator_B)


if __name__ == '__main__':
    main()
