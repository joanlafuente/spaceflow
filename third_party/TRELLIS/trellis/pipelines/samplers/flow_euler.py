from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
from torch.nn.functional import max_pool3d
import torch.nn.functional as F


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        t0 = None
        t0_high = None
        control_high = None
        control_high_mask = None
        low_control_mask = None
        control_high_lat = None
        lantent_high_control = None

        if 'control' in kwargs:
            control = kwargs['control']
            t0 = t_seq[int(kwargs['t0_idx_value'])]
            sample = noise * t0 + control * (1 - t0) # [1, 8, 16, 16, 16]

        if (kwargs.get('control_high') is not None) and (kwargs.get('local_tau_mode', None) == 'guidance'):
            control_high = kwargs['control_high']
            t0_high = t_seq[int(kwargs['t0_idx_value_high_control'])]
            control_high_mask = None
        elif (kwargs.get('control_high') is not None) and (kwargs.get('local_tau_mode', None) == 'masking'):
            control_high_mask = kwargs['control_high']
            # control_low_mask = kwargs['control_low_mask']

            # print(control_low_mask)
            # non_zero_ratio = (control_low_mask > 0).float().mean().item()
            # print(f"Initial non-zero ratio of control_low_mask: {non_zero_ratio:.4f}", flush=True)

            
            # print("Applying morphological dilation to control_low_mask... Shape:", control_low_mask.shape, flush=True)
            # control_low_mask = max_pool3d(1 - control_low_mask, kernel_size=5, stride=1, padding=2)
            # print("Dilation applied. Shape after dilation:", control_low_mask.shape, flush=True)
            # control_low_mask = 1 - control_low_mask

            lantent_high_control = kwargs['latent_high_control']
            t0_high = t_seq[int(kwargs['t0_idx_value_high_control'])]
            control_high = None
        elif (kwargs.get('control_low_mask') is not None) and (kwargs.get('local_tau_mode', None) == 'low_control_mask'):
            low_control_mask = kwargs['control_low_mask']
            non_zero_ratio = (low_control_mask > 0).float().mean().item()
            print(f"Initial non-zero ratio of low_control_mask: {non_zero_ratio:.4f}", flush=True)
            mean_value_non_zero = low_control_mask[low_control_mask > 0].mean().item()
            print(f"Mean value of non-zero elements in low_control_mask: {mean_value_non_zero:.4f}", flush=True)
            # Binarize the low_control_mask
            low_control_mask = (low_control_mask > 0).float()
            non_zero_ratio_after_binarization = (low_control_mask > 0).float().mean().item()
            print(f"Non-zero ratio of low_control_mask after binarization: {non_zero_ratio_after_binarization:.4f}", flush=True)
            print("mean value of non-zero elements in low_control_mask after binarization: {:.4f}".format(low_control_mask[low_control_mask > 0].mean().item()), flush=True)

            # Dilation of the mask
            # kernel_size = 3 # Adjust this based on how much "bleed" you want to capture
            # padding = kernel_size // 2
            
            # dilated_mask = F.max_pool3d(
            #     low_control_mask, 
            #     kernel_size=kernel_size, 
            #     stride=1, 
            #     padding=padding
            # )
            # low_control_mask = (dilated_mask > 0).float()

            t0_high = t_seq[int(kwargs['t0_idx_value_high_control'])]
            control_high_lat = kwargs.get('control_high')

            control_high_mask = None
            control_high = None
        else:
            print("No high control provided, skipping high control adjustment initialization.")
            control_high_mask = None
            control_high = None

        args = {'neg_cond': kwargs['neg_cond'],
                'cfg_strength': kwargs['cfg_strength'],
                'cfg_interval': kwargs['cfg_interval']}
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        print(f"Starting sampling with {steps} steps...", flush=True)
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            # print(f"Current t: {t:.4f}, Previous t: {t_prev:.4f}" + (f", t0: {t0:.4f}" if t0 is not None else "") 
            #         + (f", t0_high: {t0_high:.4f}" if t0_high is not None else ""), flush=True)
            local_tau_mode = kwargs.get('local_tau_mode', None)
            applying_high_guidance = (control_high is not None) and (t0_high is not None) and t > t0_high
            applying_high_mask = (control_high_mask is not None) and (t0_high is not None) and t > t0_high
            applying_low_mask_blend = (
                local_tau_mode == 'low_control_mask'
                and low_control_mask is not None
                and control_high_lat is not None
                and t0_high is not None
                and t > t0_high
            )
            if 'control' in kwargs and t0 is not None and t > t0 and not (applying_high_guidance or applying_high_mask or applying_low_mask_blend):
               print("Skipping step with control due to t > t0...", flush=True) 
               continue

            out = self.sample_once(model, sample, t, t_prev, cond, **args)
            sample = out.pred_x_prev

            if (control_high is not None)and t > t0_high:
                print("Applying high control adjustment...", flush=True)
                # sample_high = noise * t + control_high * (1 - t) # [1, 8, 16, 16, 16]
                sample = sample * (1 - kwargs['polyak_update_tau']) + control_high * kwargs['polyak_update_tau']
            elif (control_high_mask is not None) and t > t0_high:
                print("Applying high control masking...", flush=True)
                noise = torch.randn_like(sample)
                sample_gt_t = lantent_high_control # noise * t_prev + lantent_high_control * (1 - t_prev)

                polyak_low = 0.08
                polyak_high = 1.0
                sample_low_control = (1 - polyak_low) * sample * (1-control_high_mask) + polyak_low * sample_gt_t * (1-control_high_mask) # sample * (1 - control_high_mask)  
                sample_high_control = (1 - polyak_high) * sample * control_high_mask + polyak_high * lantent_high_control * control_high_mask
                sample = sample_low_control + sample_high_control
            elif applying_low_mask_blend:
                print("Applying low-control-mask high-control blend...", flush=True)
                sample_gt_t = control_high_lat # noise * t_prev + control * (1 - t_prev)

                polyak_high = kwargs['polyak_update_tau']
                sample_low_control = sample * low_control_mask
                sample_high_control = (1 - polyak_high) * sample + polyak_high * sample_gt_t
                sample = sample_low_control + sample_high_control * (1 - low_control_mask)
            else:
                print("No high control adjustment applied for this step.", flush=True)

            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample.detach()
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
