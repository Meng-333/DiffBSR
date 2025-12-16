import numpy as np
import torch
import torch.nn as nn
import math
from skimage import color as sc
import cv2
from skimage.measure import compare_mse
import torchvision.transforms as transforms
import lpips
import matplotlib.pyplot as plt




####################
# metric
####################
def calc_psnr_ssim(img1, img2, crop_border, test_Y=True):
    #
    # img1 = img1 / 255.
    # img2 = img2 / 255.
    h, w = img1.shape[:2]

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = sc.rgb2ycbcr(img1)[:, :, 0]
        im2_in = sc.rgb2ycbcr(img2)[:, :, 0]
    else:
        im1_in = img1
        im2_in = img2
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[
            crop_border : h - crop_border, crop_border : w - crop_border, :
        ]
        cropped_im2 = im2_in[
            crop_border : h - crop_border, crop_border : w - crop_border, :
        ]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[
            crop_border : h - crop_border, crop_border : w - crop_border
        ]
        cropped_im2 = im2_in[
            crop_border : h - crop_border, crop_border : w - crop_border
        ]
    else:
        raise ValueError(
            "Wrong image dimension: {}. Should be 2 or 3.".format(im1_in.ndim)
        )

    psnr = calc_psnr(cropped_im1, cropped_im2)
    ssim = calc_ssim(cropped_im1, cropped_im2)
    return psnr, ssim


def calc_mse(img1, img2):
    # np.mean((img1 - img2) ** 2)
    mse = compare_mse(img1, img2)
    return mse

def calc_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calc_ssim(img1, img2):

    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")

def calc_ergas(img1, img2, scale=4):
    channel = img1.shape[2]
    mse = compare_mse(img1, img2)
    mean2 = np.mean(img1, dtype=np.float64)**2
    ergas = 100.0*np.sqrt(mse/mean2/channel)/scale
    return ergas

def calc_lpips(img1, img2, use_gpu=True):
    # model = PerceptualSimilarity.PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu)
    # d = model.forward(img1, img2, normalize=True)
    # return d.detach().item()

    transf = transforms.ToTensor()
    test_HR = transf(img2).to(torch.float32)
    test2 = transf(img1).to(torch.float32)
    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    lpips_metrc = loss_fn_alex(test_HR, test2)
    return lpips_metrc


def plot_img(imgs, mses, psnrs, ssims, ergas, lpips, save_fn, show_label=True, show=False):
    size = list(imgs[0].shape)
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, mse, psnr, ssim, erga, lpip) in enumerate(zip(axes.flatten(), imgs, mses, psnrs, ssims, ergas, lpips)):
        ax.axis('off')
        ax.set_adjustable('box')
        # print('________img.shape', img.shape)   # [256, 256, 3]
        if list(img.shape)[0] == 3:
            # Scale to 0-255
            # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            img *= 255.0
            img = img.clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
        else:
            # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
            # img = img.squeeze().clamp(0, 1).detach().numpy()
            ax.imshow(img, cmap='gray', aspect='equal')

        if show_label:
            ax.axis('on')
            if i == 0:
                ax.set_xlabel('HR image')
            elif i == 1:
                ax.set_xlabel('LR image')

            elif i == 2:
                ax.set_xlabel('Bicubic (MSE: %.5fdB)\n (PSNR: %.5fdB)\n (SSIM: %.5f)\n (ERGA: %.5f)\n (LPIP: %.5f)' % (mse, psnr, ssim, erga, lpip))
            elif i == 3:
                ax.set_xlabel('SR image (MSE: %.5fdB)\n (PSNR: %.5fdB)\n (SSIM: %.5f)\n (ERGA: %.5f)\n (LPIP: %.5f)' % (mse, psnr, ssim, erga, lpip))

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    # save figure
    plt.savefig(save_fn)
    if show:
        plt.show()
    else:
        plt.close()


