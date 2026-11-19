"""Inference speed of UFLDv2 (batch 1, fp32, no_grad, GPU-synchronised timing),
the same protocol as the speed_simple.py scripts in LaneATT, clrnet and conditional-lane-detection.

Times the network forward. Decoding the output into lane points is not included.
Run inside WSL:  conda activate lane-det && python speed_simple.py configs/culane_res18.py
Weights come from --test_model (defaults to the trained CULane model below).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils.common import get_model, merge_config

DEFAULT_IMG = os.path.expanduser('~/datasets/CULane/driver_100_30frame/05251517_0433.MP4/00000.jpg')
DEFAULT_CKPT = 'logs/20261027_191943_lr_1e-02_b_8/model_best.pth'

# options of this script; everything else goes to the repo's config parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--img', default=DEFAULT_IMG, help='CULane image used as input')
parser.add_argument('--warmup', default=50, type=int)
parser.add_argument('--iters', default=500, type=int)
speed_args, sys.argv[1:] = parser.parse_known_args()

torch.backends.cudnn.benchmark = True
args, cfg = merge_config()
net = get_model(cfg)
state_dict = torch.load(cfg.test_model or DEFAULT_CKPT, map_location='cpu')['model']
net.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}, strict=True)
net.eval()

# same preprocessing as LaneTestDataset: resize, normalise, keep the bottom train_height rows
img_transforms = transforms.Compose([
    transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
x = img_transforms(Image.open(speed_args.img).convert('RGB'))[:, -cfg.train_height:, :]
x = x.unsqueeze(0).cuda()

with torch.no_grad():
    for i in range(speed_args.warmup):
        y = net(x)
    torch.cuda.synchronize()

    t_all = []
    for i in range(speed_args.iters):
        t1 = time.perf_counter()
        y = net(x)
        torch.cuda.synchronize()
        t_all.append(time.perf_counter() - t1)

t = np.array(t_all)
print('model: UFLDv2 resnet%s, input %dx%d' % (cfg.backbone, cfg.train_width, cfg.train_height))
print('average time: %.2f ms  (%.1f fps)' % (t.mean() * 1e3, 1 / t.mean()))
print('median time:  %.2f ms  (%.1f fps)' % (np.median(t) * 1e3, 1 / np.median(t)))
print('fastest time: %.2f ms  (%.1f fps)' % (t.min() * 1e3, 1 / t.min()))
print('slowest time: %.2f ms  (%.1f fps)' % (t.max() * 1e3, 1 / t.max()))
