import functools
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image

from img_generator import FusionGenerator


def strip2clean(path):

    image = Image.open(path)

    transform_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tensor = transform_(image).unsqueeze(dim=0)

    netG = FusionGenerator(3, 3, 64,
                           norm_layer=functools.partial(nn.InstanceNorm2d,
                                                        affine=False,
                                                        track_running_stats=False),
                           use_dropout=False, n_blocks=9)
    netG.load_state_dict(torch.load("./models/FusionGenerator.pth"))
    netG.eval()

    fake = (0.5 * (netG(tensor).data + 1.0)).cpu().clone().squeeze(dim=0)
    img = ToPILImage()(fake)

    del netG

    return img
    