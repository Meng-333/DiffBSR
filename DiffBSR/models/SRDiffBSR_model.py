import functools
import math
import time
import os
from functools import partial
from inspect import isfunction

import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
from litsr.data.srmd_degrade import SRMDPreprocessing
from litsr.metrics import *
from litsr.transforms import denormalize, normalize, tensor2uint8
from torch import nn
from torch.nn import init
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from litsr.utils.moco_builder import MoCo
from litsr.models.utils import accumulate
from torch.nn import functional as F
from archs import ArchRegistry, create_net, load_or_create_net

from archs.SRDiffBSR_unet_arch import *
from models import ModelRegistry
import random


####################
# SRDiffSR Diffusion
####################


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

def get_beta_schedule(num_diffusion_timesteps, beta_schedule='linear', beta_start=0.0001, beta_end=0.02):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "cosine":
        betas = cosine_beta_schedule(num_diffusion_timesteps, s=0.008)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        scale=4
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.rrdb = RRDBNet(3, 3, 32, 8, 16, scale=scale)
        self.channels = channels
        self.image_size = image_size
        self.conditional = conditional
        self.loss_type = loss_type
        self.scale = scale
        self.sample_tqdm = True
        self.set_loss()

        if schedule_opt is not None:
            self.set_new_noise_schedule(schedule_opt)

    def set_loss(self):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32)
        betas = get_beta_schedule(
            num_diffusion_timesteps=schedule_opt['n_timestep'],
            beta_schedule=schedule_opt['schedule'],
            beta_start=schedule_opt['linear_start'],
            beta_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, noise_pred, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, img_lr_up, noise_pred=None, clip_denoised=True, repeat_noise=False):
        if noise_pred is None:
            noise_pred = self.denoise_fn(x, t, cond=cond, img_lr_up=img_lr_up)
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, noise_pred=noise_pred, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, x_in, x_lr, continous=False):
        shape = x_in.shape
        img_lr_up = x_in
        img_lr = x_lr
        rrdb_out, cond = self.rrdb(img_lr, True)

        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        it = reversed(range(0, self.num_timesteps))
        if self.sample_tqdm:
            it = tqdm(it, desc='sampling loop time step', total=self.num_timesteps)
        for i in it:
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond, img_lr_up)
        img = self.res2img(img, img_lr_up)
        # if continous:
        #     return img
        # else:
        #     return img[-1]
        return img

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), None, continous)

    @torch.no_grad()
    def super_resolution(self, x_in, x_lr, continous=False):
        return self.p_sample_loop(x_in, x_lr, continous)

    @torch.no_grad()
    def interpolate(self, x1, x2, img_lr, img_lr_up, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        use_rrdb = True
        if use_rrdb:
            rrdb_out, cond = self.rrdb(img_lr, True)
        else:
            cond = img_lr

        assert x1.shape == x2.shape

        x1 = self.img2res(x1, img_lr_up)
        x2 = self.img2res(x2, img_lr_up)

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img, x_recon = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond, img_lr_up)

        img = self.res2img(img, img_lr_up)
        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        t_cond = (t[:, None, None, None] >= 0).float()
        t = t.clamp_min(0)
        return (
                       extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                       extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
               ) * t_cond + x_start * (1 - t_cond)

    def p_losses(self, x_in, noise=None):

        # self, x_start, t, cond, img_lr_up, noise=None
        x_hr = x_in['HR']
        img_lr_up = x_in['SR']
        img_lr = x_in['LR']
        x_start = self.img2res(x_hr, img_lr_up)

        rrdb_out, cond = self.rrdb(img_lr, True)
        [b, c, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_tp1_gt = self.q_sample(x_start=x_start, t=t, noise=noise)

        noise_pred = self.denoise_fn(x_tp1_gt, t, cond, img_lr_up)
        # noise_pred = self.denoise_fn(torch.cat([x_in['SR'], x_tp1_gt], dim=1), t, cond, img_lr_up)

        if self.loss_type == 'l1':
            # loss = self.loss_func(noise, noise_pred)
            loss = (noise - noise_pred).abs().mean()
        else:
            raise NotImplementedError()
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

    def res2img(self, img_, img_lr_up, clip_input=None):
        if clip_input is None:
            clip_input = True
        if clip_input:
            img_ = img_.clamp(-1, 1)
        img_ = img_ / 2.0 + img_lr_up
        return img_

    def img2res(self, x, img_lr_up, clip_input=None):
        if clip_input is None:
            clip_input = True
        x = (x - img_lr_up) * 2.0
        if clip_input:
            x = x.clamp(-1, 1)
        return x



####################
# Lightning Model
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="kaiming", scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print("Initialization method [{:s}]".format(init_type))
    if init_type == "normal":
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == "kaiming":
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            "initialization method [{:s}] not implemented".format(init_type)
        )


@ModelRegistry.register()
class SRDiffBlindSRModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(opt)
        self.opt = opt.lit_model.args

        self.pretrain_epochs = self.hparams.trainer.pretrain_epochs

        # init super-resolution network
        self.encoder = load_or_create_net(self.opt.encoder)
        self.dg_encoder = ArchRegistry.get(self.opt.dg_encoder)
        self.E = MoCo(base_encoder=self.dg_encoder)
        self.model = create_net(self.opt.diffusion_model)
        init_weights(self.model, init_type="orthogonal")

        if self.opt.get("ema", False):
            self.ema = create_net(self.opt.diffusion_model)  # U-Net
            init_weights(self.ema, init_type="orthogonal")
        else:
            self.ema = None

        # loss weight
        self.contrast_loss_weight = self.opt.get("contrast_loss_weight", 0.01)

        self.diffusion = GaussianDiffusion(
            self.model,
            image_size=self.opt.diffusion.image_size,
            channels=self.opt.diffusion.channels,
            loss_type=self.opt.diffusion.loss_type,
            conditional=self.opt.diffusion.conditional,
            schedule_opt=self.opt.beta_schedule,
            scale=self.opt.valid.scale
        )

        self.mean, self.std = self.opt.mean, self.opt.std
        self.rgb_range = self.hparams.data_module.args.rgb_range
        self.normalize = lambda tensor: normalize(
            tensor, self.mean, self.std, inplace=True
        )
        self.denormalize = lambda tensor: denormalize(
            tensor, self.mean, self.std, inplace=True
        )

        theta_ = random.randint(0, 180)

        self.degrade = SRMDPreprocessing(
            self.opt.scale,
            kernel_size=self.opt.blur_kernel,
            blur_type=self.opt.blur_type,
            sig_min=self.opt.sig_min,
            sig_max=self.opt.sig_max,
            lambda_min=self.opt.lambda_min,
            lambda_max=self.opt.lambda_max,
            theta=theta_,
            noise=self.opt.noise,
        )
        self.valid_degrade = SRMDPreprocessing(
            self.opt.scale,
            kernel_size=self.opt.valid.blur_kernel,
            blur_type=self.opt.valid.blur_type,
            sig=self.opt.valid.get("sig"),
            lambda_1=self.opt.valid.get("lambda_1"),
            lambda_2=self.opt.valid.get("lambda_2"),
            theta=self.opt.valid.get("theta"),
            noise=self.opt.valid.get("noise"),
        )

        self.contrast_loss = th.nn.CrossEntropyLoss()

    def training_step(self, batch):
        hr = batch
        hr.mul_(255.0)
        lr, _ = self.degrade(hr)
        hr.div_(255.0)
        lr.div_(255.0)
        self.normalize(hr)
        self.normalize(lr)

        im_q = lr[:, 0, ...]
        im_k = lr[:, 1, ...]
        hr = hr[:, 0, ...]

        b, c, h, w = hr.shape

        if self.current_epoch < self.pretrain_epochs:
            _, output, target = self.E(im_q, im_k)
            loss_contrast = self.contrast_loss_weight * self.contrast_loss(output, target)

            hr_ = self.encoder(im_q)
            loss_consis = F.l1_loss(hr_, hr)

            loss = loss_contrast + loss_consis
            loss_elbo = 0
        else:
            fea, output, target = self.E(im_q, im_k)  # estimator loss
            loss_contrast = self.contrast_loss_weight * self.contrast_loss(output, target)

            hr_ = self.encoder(im_q)  # rrdbnet loss
            loss_consis = F.l1_loss(hr_, hr)

            l_pix = self.diffusion({"HR": hr, "SR": hr_, "LR": im_q})
            loss_elbo = l_pix.sum() / int(b * c * h * w)  # noise loss

            loss = loss_contrast + loss_consis + loss_elbo

            if self.ema:
                accumulate(
                    self.ema,
                    self.model.module
                    if isinstance(self.model, th.nn.DataParallel)
                    else self.model,
                    self.opt.ema_rate,
                )

        self.log("train/loss_contrast", loss_contrast)
        self.log("train/loss_consis", loss_consis)
        self.log("train/loss_elbo", loss_elbo)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        # choose use ema or not
        _model = self.ema if self.ema else self.model

        hr, name = batch
        if len(hr.shape) == 4:
            hr = hr.unsqueeze(1)
        hr.mul_(255.0)
        lr, _ = self.valid_degrade(hr, random=False)
        lr = lr.squeeze(0)

        Bicubic_interpolation = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)
        bic = Bicubic_interpolation(lr)

        bic.div_(255.0)
        self.normalize(bic)

        hr.div_(255.0)
        lr.div_(255.0)
        self.normalize(hr)
        self.normalize(lr)

        lr, hr, bic = lr.squeeze(1), hr.squeeze(1), bic.squeeze(1),

        th.cuda.synchronize()
        tic = time.time()

        th.cuda.synchronize()
        toc = time.time()

        conditional_img = self.encoder(lr)
        sr = self.diffusion.super_resolution(conditional_img, lr)

        self.denormalize(sr)
        self.denormalize(hr)
        self.denormalize(lr)
        self.denormalize(bic)

        sr_np, hr_np, lr_np, bic_np = tensor2uint8(
            [sr.cpu()[0], hr.cpu()[0], lr.cpu()[0], bic.cpu()[0]], self.rgb_range
        )

        bmse = calc_mse(bic_np, hr_np)
        bpsnr = calc_psnr(bic_np, hr_np)
        bssim = calc_ssim(bic_np, hr_np)
        bergas = calc_ergas(bic_np, hr_np)
        blpips = calc_lpips(bic_np, hr_np)

        mse = calc_mse(sr_np, hr_np)
        psnr = calc_psnr(sr_np, hr_np)
        ssim = calc_ssim(sr_np, hr_np)
        ergas = calc_ergas(sr_np, hr_np)
        lpips = calc_lpips(sr_np, hr_np)

        result_imgs = [hr_np, lr_np, bic_np, sr_np]
        mses = [None, None, bmse, mse]
        psnrs = [None, None, bpsnr, psnr]
        ssims = [None, None, bssim, ssim]
        ergass = [None, None, bergas, ergas]
        lpipss = [None, None, blpips, lpips]

        plot_np = (result_imgs, mses, psnrs, ssims, ergass, lpipss)

        return {
            "val_bmse": bmse,
            "val_bpsnr": bpsnr,
            "val_bssim": bssim,
            "val_bergas": bergas,
            "val_blpips": blpips,
            "val_mse": mse,
            "val_psnr": psnr,
            "val_ssim": ssim,
            "val_ergas": ergas,
            "val_lpips": lpips,
            "log_img_sr": sr_np,
            "log_img_lr": lr_np,
            "log_img_plot": plot_np,
            "name": name[0],
            "time": toc - tic,
        }

    def validation_epoch_end(self, outputs):
        # avg_val_loss = th.stack([x["val_loss"] for x in outputs]).mean()
        # self.log("val/loss", avg_val_loss, on_epoch=True, prog_bar=True, logger=True)

        avg_val_mse = np.array([x["val_mse"] for x in outputs]).mean()
        avg_val_ssim = np.array([x["val_ssim"] for x in outputs]).mean()
        avg_val_ergas = np.array([x["val_ergas"] for x in outputs]).mean()
        avg_val_lpips = np.array([x["val_lpips"] for x in outputs]).mean()

        avg_val_psnr = np.array([x["val_psnr"] for x in outputs]).mean()
        self.log("val/psnr", avg_val_psnr, on_epoch=True, prog_bar=True, logger=True)

        # self.log("val/lpips", avg_val_lpips, on_epoch=True, prog_bar=True, logger=True)

        f = open('./logs/srdiffbsr_aniso/validation_epoch_end.txt', 'a')
        print_metrics = "--- current_epoch:{:d} MSE: {:.5f}; PSNR: {:.5f}; SSIM: {:.5f}; ERGAS: {:.5f}; LPIPS: {:.5f}".format(
            self.current_epoch, avg_val_mse, avg_val_psnr, avg_val_ssim, avg_val_ergas, avg_val_lpips)
        f.write(print_metrics + os.linesep)
        f.close()

        log_img_sr = outputs[0]["log_img_sr"]

        self.logger.experiment.add_image(
            "img_sr", log_img_sr, self.global_step, dataformats="HWC"
        )

        return

    def test_step(self, batch, *args, **kwargs):
        return self.validation_step(batch, *args, **kwargs)

    def test_step_lr_only(self, batch, *args):
        # choose use ema or not
        _model = self.ema if self.ema else self.model

        lr, name = batch
        self.normalize(lr)

        th.cuda.synchronize()
        tic = time.time()

        conditional_img = self.encoder(lr)
        sr = self.diffusion.super_resolution(conditional_img)

        th.cuda.synchronize()
        toc = time.time()
        self.denormalize(sr)
        self.denormalize(lr)

        [sr_np, lr_np] = tensor2uint8([sr.cpu()[0], lr.cpu()[0]], self.rgb_range)

        return {
            "log_img_sr": sr_np,
            "log_img_lr": lr_np,
            "name": name[0],
            "time": toc - tic,
        }

    def test_step_lr_hr_paired(self, batch, *args, **kwargs):
        # choose use ema or not
        _model = self.ema if self.ema else self.model
        lr, hr, name = batch
        self.normalize(hr)
        self.normalize(lr)

        th.cuda.synchronize()
        tic = time.time()

        th.cuda.synchronize()
        toc = time.time()

        conditional_img = self.encoder(lr)
        sr = self.diffusion.super_resolution(conditional_img)

        self.denormalize(sr)
        self.denormalize(hr)
        self.denormalize(lr)

        crop_border = int(np.ceil(float(hr.shape[2]) / lr.shape[2]))
        sr_np, hr_np, lr_np = tensor2uint8(
            [sr.cpu()[0], hr.cpu()[0], lr.cpu()[0]], self.rgb_range
        )
        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=crop_border, test_Y=self.opt.valid.test_Y
        )
        return {
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "log_img_lr": lr_np,
            "name": name[0],
            "time": toc - tic,
        }

    def configure_optimizers(self):
        betas = self.opt.optimizer.get("betas") or (0.9, 0.999)
        optimizer = th.optim.Adam(
            self.parameters(), lr=self.opt.optimizer.lr, betas=betas
        )
        if self.opt.optimizer.get("lr_scheduler_step"):
            LR_scheduler = th.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.opt.optimizer.lr_scheduler_step,
                gamma=self.opt.optimizer.lr_scheduler_gamma,
            )
        elif self.opt.optimizer.get("lr_scheduler_milestones"):
            LR_scheduler = th.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.opt.optimizer.lr_scheduler_milestones,
                gamma=self.opt.optimizer.lr_scheduler_gamma,
            )
        else:
            raise Exception("No lr settings found! ")
        return [optimizer], [LR_scheduler]
