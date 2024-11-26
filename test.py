import torch
from PIL import Image
import torchvision.transforms as transform
from train import Generator

input_path = 'D:/pythonProject/cycleGAN_styleTransfer/123.jpg'
output_path = 'D:/pythonProject/cycleGAN_styleTransfer/new123.jpg'
img_size = 256


def test(input_path):
    # 加载图片
    input_image = Image.open(input_path)
    transforms = transform.Compose([
        transform.ToTensor(),
        transform.Resize((img_size, img_size)),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_image = transforms(input_image.copy())
    input_image = input_image.unsqueeze(0).to('cuda')  # 增加batch维度

    # 加载模型
    cycle_gan = torch.load('D:/pythonProject/cycleGAN_styleTransfer/cycleGAN_model.pth')
    # 实例化生成器
    generator = Generator().to('cuda')
    generator.load_state_dict(cycle_gan['generator_B'], strict=False)

    # 生成图片
    output_image = generator(input_image)
    output_image = output_image.squeeze(0)  # 移除batch维度
    # output_image = output_image.permute(1, 2, 0)  # 转换为HWC格式
    output_image = transform.ToPILImage()(output_image)  # 转换为PIL图像
    output_image.save(output_path)


if __name__ == '__main__':
    test(input_path)
