import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from options.test import args_test
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from load_data.data import get_training_set, get_eval_set
from torch.utils.data import DataLoader
from module.base_module import PSNR, save_images, print_network, calculate_mse, calculate_psnr, calculate_ssim, calculate_ergas, calculate_lpips, plot_img, save_img
from module.blindsr import BlindSR as Net
from SimCLR.my_nt_xent import NT_Xent
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
from tensorboardX import SummaryWriter
import numpy as np
import cv2
from thop import profile
import imageio

gpus_list = range(args_test.n_GPUs)
hostname = str(socket.gethostname())

print('===> Building model ')
model = Net(args_test)
print('---------- Networks architecture -------------')
# print_network(model)

model_name = '/root/autodl-tmp/DRSR/runs_AIRS_WHU/checkpoint/best_4x_blindsr_epoch_593.pth'
save_results_dir = '/root/autodl-tmp/DRSR/runs_infer'

if os.path.exists(model_name):
    model.load_state_dict({k.replace('module.', ''): v for k, v in
                           torch.load(model_name, map_location=lambda storage, loc: storage).items()})
    print('pretrained model is loaded!')
else:
    raise Exception('No such pretrained model!')

# To GPU
model = model.cuda(gpus_list[0])

############### infer #################
def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor

def infer():
    model.eval()
    # load data
    img_fn = '/root/autodl-tmp/DRSR/data/infer/img'
    imgs = os.listdir(img_fn)
    for img_ in imgs:
        img_dir = os.path.join(img_fn, img_)
        img = imageio.imread(img_dir)

        lr = np2Tensor(img)
        lr = lr.unsqueeze(0)

        with torch.no_grad():
            lr = Variable(lr).cuda(gpus_list[0])
            prediction = model(lr, lr)

        print('===>Processing: %s' % img_)
        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)

        # save imges
        sr_folder = save_results_dir + '/infer/'
        save_img(prediction, sr_folder, img_)

##  Infer Start!!!!
infer()



