
'''
 @FileName    : interhuman_diffusion.py
 @EditTime    : 2023-10-14 19:05:18
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils.imutils import cam_crop2full, vis_img
from utils.geometry import perspective_projection
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix

from model.utils import *
from model.blocks import *
from model.diffusion_posenet import PoseNet
from model.diffusion_segnet import segnet
from model.diffusion_poseseg import Res_catconv
import cv2
from utils.imutils import *


class interhuman_diffusion(nn.Module):
    def __init__(self, smpl, num_joints=21, latentD=32, frame_length=16, n_layers=1, hidden_size=256, bidirectional=True,):
        super(interhuman_diffusion, self).__init__()
        self.smpl = smpl

        num_frame = frame_length
        num_agent = 2
        self.eval_initialized = False
        num_timesteps = 100
        beta_scheduler = 'cosine'
        self.timestep_respacing = 'ddim5'

        # Use float64 for accuracy.
        betas = get_named_beta_schedule(beta_scheduler, num_timesteps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.sampler = UniformSampler(num_timesteps)
        pose = PoseNet()
        seg = segnet()
        self.poseseg = Res_catconv(seg, pose)

        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1, dilation=1)
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1, dilation=1)


    def init_eval(self,):
    
        use_timesteps = set(space_timesteps(self.num_timesteps, self.timestep_respacing))
        self.timestep_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        self.test_betas = np.array(new_betas)

        self.num_timesteps_test = int(self.test_betas.shape[0])

        test_alphas = 1.0 - self.test_betas
        self.test_alphas_cumprod = np.cumprod(test_alphas, axis=0)
        self.test_alphas_cumprod_prev = np.append(1.0, self.test_alphas_cumprod[:-1])
        self.test_alphas_cumprod_next = np.append(self.test_alphas_cumprod[1:], 0.0)
        assert self.test_alphas_cumprod_prev.shape == (self.num_timesteps_test,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.testsqrt_alphas_cumprod = np.sqrt(self.test_alphas_cumprod)
        self.test_sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.test_alphas_cumprod)
        self.test_log_one_minus_alphas_cumprod = np.log(1.0 - self.test_alphas_cumprod)
        self.test_sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.test_alphas_cumprod)
        self.test_sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.test_alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.test_posterior_variance = (
                self.test_betas * (1.0 - self.test_alphas_cumprod_prev) / (1.0 - self.test_alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.test_posterior_log_variance_clipped = np.log(
            np.append(self.test_posterior_variance[1], self.test_posterior_variance[1:])
        )
        self.test_posterior_mean_coef1 = (
                self.test_betas * np.sqrt(self.test_alphas_cumprod_prev) / (1.0 - self.test_alphas_cumprod)
        )
        self.test_posterior_mean_coef2 = (
                (1.0 - self.test_alphas_cumprod_prev)
                * np.sqrt(test_alphas)
                / (1.0 - self.test_alphas_cumprod)
        )


    def condition_process(self, data):
        img_info = {}
        cond = data['img']
        return cond

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def generate_noise(self, init_pose, noise=None):
        if noise is None:
            noise = torch.randn((init_pose.shape[0], init_pose.shape[1], init_pose.shape[2], init_pose.shape[3]), device=init_pose[-1].device, dtype=init_pose[-1].dtype)
        
        return noise

    def trans2cam(self, trans, img_info):
        
        
        img_h, img_w = img_info['full_img_shape'][:, 0], img_info['full_img_shape'][:, 1]
        cx, cy, b = img_info['center'][:, 0], img_info['center'][:, 1], img_info['scale'] * 200
        w_2, h_2 = img_w / 2., img_h / 2.

        cam_z = (2 * img_info['focal_length']) / (b * trans[:,2] + 1e-9)

        bs = b * cam_z + 1e-9

        cam_x = trans[:,0] - (2 * (cx - w_2) / bs)
        cam_y = trans[:,1] - (2 * (cy - h_2) / bs)

        cam = torch.stack([cam_z, cam_x, cam_y], dim=-1)

        return cam

    def input_process(self, data):


        x_start = data['gt_heat'] # data['gt_heatmap']
        return x_start

    def inference(self, x_t, t, cond, data):
        img = cond
        # Network forward


        pred = self.poseseg(x_t, t, cond, data)




        return pred

    # def visualize(self, x_t, data, img_info, t_idx):
    def visualize(self, x_t, cond):
        mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
        std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
        for x, img in zip(x_t, cond):
            img = (img.detach().cpu().numpy().transpose(1,2,0) * std) + mean
            # vis_img('img', img)
            img_nummpy = img * 255
            vis_x_0 = x
            vis_x_0 = vis_x_0.detach().cpu().numpy().astype(np.float32)
            vis_x_0 = vis_x_0.transpose(1, 2, 0)
            vis_x_0 = np.max(vis_x_0, axis=2)  
            gtt = convert_color(vis_x_0*255)
            # vis_img('heat', gtt)
            dst = cv2.addWeighted(gtt,0.5, img_nummpy.astype(np.uint8),0.5,0)
            # name = f"output_{i}.jpg"
            vis_img('img', dst)
        # cv2.imwrite("name.png", dst)


    def visualize_sampling(self, x_start, ts, data, img_info, mean, noise):

        device, dtype = ts.device, ts.dtype
        indices = list(range(self.num_timesteps))[::-1]

        for t in indices:
            t_idx = t
            t = torch.from_numpy(np.array([t] * x_start.shape[0])).to(device=device, dtype=dtype)

            x_t = self.q_sample(x_start, t, noise=noise)

            x_t = x_t + mean

            pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
            shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
            pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

            self.visualize(pose, shape, pred_cam, data, img_info, t_idx)

    def forward(self, data):
        batch_size = data['img'].shape[0]

        cond = self.condition_process(data)

        init_pose = data['gt_heat'][-1]  # data['init_heatmap']
        noise = self.generate_noise(init_pose)

        if self.training:
            
            x_start = self.input_process(data)

            t, _ = self.sampler.sample(batch_size, x_start[-1].device)

            x_t = self.q_sample(x_start[-1], t, noise=noise)

            vis = False
            if vis:
                self.visualize(x_t, cond)

            pred = self.inference(x_t, t, cond, data)

        else:
            if not self.eval_initialized:
                self.init_eval()
                self.eval_initialized = True
                
            pred = self.ddim_sample_loop(noise, cond, data)
    

        return pred

    def ddim_sample_loop(self, noise, cond, data, eta=0.0):
        indices = list(range(self.num_timesteps_test))[::-1]

        img = noise
        preds = []
        for i in indices:
            t = torch.tensor([i] * noise.shape[0], device=noise.device)
            pred = self.ddim_sample(img, t, cond, data)
            preds.append(pred)


            model_output = pred['preheat'][-1]
            model_output = model_output.reshape(*img.shape)
            model_output = model_output
            vis = False
            if vis:
                self.visualize(model_output, data)
            model_variance, model_log_variance = (
                    self.test_posterior_variance,
                    self.test_posterior_log_variance_clipped,
                )
            
            model_variance = extract_into_tensor(model_variance, t, img.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, img.shape)

            pred_xstart = model_output

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=img, t=t
            )

            assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == img.shape
            )

            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(img, t, pred_xstart)

            alpha_bar = extract_into_tensor(self.test_alphas_cumprod, t, img.shape)
            alpha_bar_prev = extract_into_tensor(self.test_alphas_cumprod_prev, t, img.shape)
            sigma = (
                    eta
                    * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            # Equation 12.
            noise = self.generate_noise(img) # data['init_heatmap']
            mean_pred = (
                    pred_xstart * torch.sqrt(alpha_bar_prev)
                    + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            )
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            )  # no noise when t == 0
            sample = mean_pred + nonzero_mask * sigma * noise

            img = sample
            vis = False
            if vis:
                self.visualize(img, data, i)
        return preds[-1]


    def ddim_sample(self, x, ts, cond, data):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        pred = self.inference(x, new_ts, cond, data)

        return pred

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                extract_into_tensor(self.test_posterior_mean_coef1, t, x_t.shape) * x_start
                + extract_into_tensor(self.test_posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.test_posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.test_posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                extract_into_tensor(self.test_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / extract_into_tensor(self.test_sqrt_recipm1_alphas_cumprod, t, x_t.shape)


if __name__ == '__main__':
    model = interhuman_diffusion(None)

    data = {'features': torch.rand((1, 16, 2, 1024)), 
            'pose_6d': torch.rand((1, 16, 2, 144)), 
            'betas': torch.rand((1, 16, 2, 10)), 
            'gt_cam_t': torch.rand((1, 16, 2, 3)),
            'img': torch.rand((1, 3, 256, 256)),
            'fullheat':torch.rand((1, 17, 256, 256)), 
            'fullheatmap':torch.rand((1, 17, 256, 256)),
            'mask':torch.rand((3, 256, 256))
              }

    pred = model(data)

    model.eval()
    pred = model(data)
