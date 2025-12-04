import argparse
import sys

from thop import profile
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from data import *
from metrics import psnr, ssim


from model import Generator


def eval(opt):
    dataset_test = MyValueDataSet(opt.dataset)
    val_loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

    model = Generator().to(opt.device)
    model.load_state_dict(torch.load(opt.model_dir))

    params = sum([param.numel() for param in model.parameters()])
    params_m = params / 1000000
    print(f'params:{params_m:.2f}M')

    input_tensor = torch.randn(1, 3, 256, 256).to(opt.device)
    flops, _ = profile(model, inputs=(input_tensor,))
    flops_g = flops / 1000000000
    print(f'FLOPs:{flops_g:.2f}G')


    psnr_val = []
    ssim_val = []
    with torch.no_grad():
        model.eval()

        for image, target, img_name in tqdm(val_loader, file=sys.stdout):
            image, target = image.to(opt.device), target.to(opt.device)

            pred = model(image)
            pred = pred[2]

            if opt.save_image:
                save_path = output_path + img_name[0]
                save_image(pred, save_path)

            psnr1 = psnr(pred, target)
            ssim1 = ssim(pred, target).item()
            psnr_val.append(psnr1)
            ssim_val.append(ssim1)

    avg_psnr = np.mean(psnr_val)
    avg_ssim = np.mean(ssim_val)
    print(f"PSNR:{avg_psnr:.4f} SSIM:{avg_ssim:.4f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help='GPU name')
    parser.add_argument("--save_image", type=bool, default=False, help="whether save test images")
    parser.add_argument("--dataset", type=str, default='E:/smr/datasets/Haze4K', help="dataset path")
    parser.add_argument("--model_dir", type=str, default='./checkpoint/Haze4K.pth', help="path of pth file")
    opt = parser.parse_args()

    output_path = os.path.join('test/')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    eval(opt)
