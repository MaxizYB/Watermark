"""
FDW Pipeline: Modified Stable Diffusion pipeline with FDSC hook.
Inherits from GS's InversableStableDiffusionPipeline and adds
Frequency-Domain Soft Constraint (FDSC) during the denoising loop.
"""

import copy
import torch
import numpy as np
from typing import Callable, List, Optional, Union
from functools import partial

try:
    from transformers import CLIPImageProcessor as CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
except ImportError:
    from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline
from diffusers.utils import BaseOutput
import PIL

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Gaussian-Shading'))
from modified_stable_diffusion import ModifiedStableDiffusionPipelineOutput
from inverse_stable_diffusion import InversableStableDiffusionPipeline, backward_ddim

# ── diffusers ≥ 0.25 compatibility patch ─────────────────────────────────────
# In diffusers 0.37+, StableDiffusionPipeline.__init__ has a new `image_encoder`
# parameter at position 8. GS's subclasses call super().__init__ with positional
# args, so `requires_safety_checker` (bool) lands in the `image_encoder` slot and
# crashes register_modules. We monkey-patch both GS classes to use keyword args.
import inspect as _inspect
_sd_params = list(_inspect.signature(StableDiffusionPipeline.__init__).parameters)

if 'image_encoder' in _sd_params:
    from modified_stable_diffusion import ModifiedStableDiffusionPipeline
    from inverse_stable_diffusion import InversableStableDiffusionPipeline as _BaseInversable
    from functools import partial as _partial

    def _patched_mod_init(self, vae, text_encoder, tokenizer, unet, scheduler,
                          safety_checker, feature_extractor,
                          image_encoder=None, requires_safety_checker=False):
        # Call StableDiffusionPipeline.__init__ directly with keyword args
        StableDiffusionPipeline.__init__(
            self,
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=safety_checker,
            feature_extractor=feature_extractor, image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

    def _patched_inv_init(self, vae, text_encoder, tokenizer, unet, scheduler,
                          safety_checker, feature_extractor,
                          image_encoder=None, requires_safety_checker=False):
        StableDiffusionPipeline.__init__(
            self,
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=safety_checker,
            feature_extractor=feature_extractor, image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        from functools import partial
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)
        self.count = 0

    ModifiedStableDiffusionPipeline.__init__ = _patched_mod_init
    _BaseInversable.__init__ = _patched_inv_init


def _apply_fdsc(z_t: torch.Tensor,
                W_freq_fft: torch.Tensor,
                t_ratio: float,
                t_start: float,
                t_end: float,
                alpha_max: float) -> torch.Tensor:
    """
    Apply Frequency-Domain Soft Constraint to latent z_t.
    Only active when t_start <= t_ratio <= t_end.
    """
    if not (t_start <= t_ratio <= t_end):
        return z_t

    # Linearly decay alpha as t decreases (early steps have more noise → larger push)
    alpha = alpha_max * (t_ratio - t_start) / (t_end - t_start + 1e-8)

    z_f = z_t.float()
    Z = torch.fft.fft2(z_f)
    Z_new = Z + alpha * W_freq_fft.to(Z.device)
    z_new = torch.fft.ifft2(Z_new).real

    return z_new.to(z_t.dtype)


def _inject_template_on_zt0(z_t0: torch.Tensor,
                             template_mask: torch.Tensor,
                             template_pattern: torch.Tensor,
                             gamma: float) -> torch.Tensor:
    """
    MaXsive Eq.(8): inject X-template on predicted x_0 (z_t0) in Fourier domain.
    z'_t0 = F^{-1}( F(z_t0) + M * gamma * pattern * std(|F(z_t0)|) )
    Uses fftshift so template centered at (H//2, W//2) aligns with DC.
    """
    z_f = z_t0.float()
    Z = torch.fft.fftshift(torch.fft.fft2(z_f), dim=(-2, -1))
    std_val = Z.abs().std().clamp_min(1e-6)
    injection = gamma * template_mask.to(Z.device) * template_pattern.to(Z.device) * std_val
    injected = Z + injection
    z_new = torch.fft.ifft2(torch.fft.ifftshift(injected, dim=(-2, -1))).real
    max_val = z_f.abs().max().clamp_min(1.0) * 2.0
    z_new = z_new.clamp(-max_val, max_val)
    return z_new.to(z_t0.dtype)


def _inject_template_shallow(z_t: torch.Tensor,
                              template_mask: torch.Tensor,
                              template_pattern: torch.Tensor,
                              gamma: float) -> torch.Tensor:
    """
    ShallowDiffuse-style injection: add template directly on z_t (current latent)
    at a single timestep. The watermark lies in the null space of the low-rank
    Jacobian at t*=0.3T, preserving image quality while remaining detectable.
    
    Same Fourier-domain mechanism as _inject_template_on_zt0 but applied to z_t
    instead of predicted x_0. Used before UNet forward pass.
    """
    z_f = z_t.float()
    Z = torch.fft.fftshift(torch.fft.fft2(z_f), dim=(-2, -1))
    std_val = Z.abs().std().clamp_min(1e-6)
    injection = gamma * template_mask.to(Z.device) * template_pattern.to(Z.device) * std_val
    injected = Z + injection
    z_new = torch.fft.ifft2(torch.fft.ifftshift(injected, dim=(-2, -1))).real
    max_val = z_f.abs().max().clamp_min(1.0) * 3.0
    z_new = z_new.clamp(-max_val, max_val)
    return z_new.to(z_t.dtype)


def _inject_pixel_template(image_np: np.ndarray,
                            pixel_template_mask: np.ndarray,
                            gamma: float) -> np.ndarray:
    """
    Inject X-template directly into pixel-space image via Fourier domain.
    image_np: [H, W, 3] float32 in [0,1].
    pixel_template_mask: [H, W] binary mask at template positions (fftshift coords).
    gamma: injection strength (relative to std of Fourier magnitude).

    Returns modified image in [0,1].
    """
    out = image_np.copy()
    for c in range(image_np.shape[2]):
        ch = image_np[:, :, c].astype(np.float64)
        F = np.fft.fftshift(np.fft.fft2(ch))
        std_val = np.abs(F).std()
        # Inject real-valued perturbation at template positions
        F_new = F + pixel_template_mask * gamma * std_val
        ch_new = np.fft.ifft2(np.fft.ifftshift(F_new)).real
        out[:, :, c] = ch_new
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def build_pixel_template_mask(H: int, W: int, base_angle: float = 45.0,
                               r_fracs=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)) -> np.ndarray:
    """
    Build X-template mask in pixel-space Fourier domain (fftshift coordinates).
    Returns [H, W] binary mask.
    """
    mask = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    for arm_offset in (0.0, 90.0):
        arm_rad = np.deg2rad(base_angle + arm_offset)
        for sign in (1.0, -1.0):
            for r_frac in r_fracs:
                r = r_frac * min(H, W) / 2
                y = int(round(cy + sign * r * np.sin(arm_rad)))
                x = int(round(cx + sign * r * np.cos(arm_rad)))
                if 0 <= y < H and 0 <= x < W:
                    mask[y, x] = 1.0
    return mask


class FDWStableDiffusionPipeline(InversableStableDiffusionPipeline):
    """
    Extends InversableStableDiffusionPipeline with FDSC injection.
    Usage is identical to the GS pipeline; pass extra kwargs:
        fdsc_template_fft : pre-computed FFT of watermark template (from FDW_Watermark.get_fdsc_template())
        fdsc_t_start      : fraction of steps to start FDSC (default 0.2)
        fdsc_t_end        : fraction of steps to end FDSC (default 0.6)
        fdsc_alpha_max    : max perturbation strength (default 0.015)
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator=None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable] = None,
        callback_steps: Optional[int] = 1,
        # FDSC-specific
        fdsc_template_fft: Optional[torch.Tensor] = None,
        fdsc_t_start: float = 0.2,
        fdsc_t_end: float = 0.6,
        fdsc_alpha_max: float = 0.015,
        # ShallowDiffuse / MaXsive-style template injection (latent z_t0)
        template_mask: Optional[torch.Tensor] = None,
        template_pattern: Optional[torch.Tensor] = None,
        template_t_start: float = 0.25,
        template_t_end: float = 0.40,
        template_gamma: float = 0.015,
        template_mode: str = 'shallow',
        # Pixel-space template injection (post-decode, for geometric correction)
        pixel_template_gamma: float = 0.0,
        **kwargs,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_cfg = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_cfg, negative_prompt
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        T_total = len(timesteps)

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        init_latents = copy.deepcopy(latents)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        shallow_injected = False
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                t_ratio = t.item() / self.scheduler.config.num_train_timesteps

                # ── ShallowDiffuse: inject on z_t BEFORE UNet at t*=0.3T ──
                if (template_mask is not None and template_pattern is not None
                        and template_mode == 'shallow' and not shallow_injected
                        and abs(t_ratio - template_t_start) < 0.02):
                    latents = _inject_template_shallow(
                        latents, template_mask, template_pattern, template_gamma
                    )
                    shallow_injected = True

                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                scheduler_out = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                prev_sample = scheduler_out.prev_sample

                # ── Template injection on z_t0 (MaXsive Eq.8) ──────────────
                if (template_mask is not None and template_pattern is not None
                        and template_mode == 'maxsive'):
                    t_ratio = t.item() / self.scheduler.config.num_train_timesteps
                    if template_t_start <= t_ratio <= template_t_end:
                        pred_orig = scheduler_out.pred_original_sample
                        if pred_orig is None:
                            alpha_t = self.scheduler.alphas_cumprod[t.item()].to(device=latents.device)
                            pred_orig = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                        pred_orig_new = _inject_template_on_zt0(
                            pred_orig, template_mask, template_pattern, template_gamma
                        )
                        step_size = self.scheduler.config.num_train_timesteps // num_inference_steps
                        prev_t = t.item() - step_size
                        if prev_t >= 0:
                            alpha_prev = self.scheduler.alphas_cumprod[prev_t].to(device=latents.device)
                        else:
                            alpha_prev = torch.tensor(1.0, device=latents.device)
                        prev_sample = prev_sample + alpha_prev.sqrt() * (pred_orig_new - pred_orig)

                latents = prev_sample

                # ── FDSC injection ──────────────────────────────────────────
                if fdsc_template_fft is not None:
                    t_ratio = t.item() / self.scheduler.config.num_train_timesteps
                    latents = _apply_fdsc(
                        latents, fdsc_template_fft,
                        t_ratio, fdsc_t_start, fdsc_t_end, fdsc_alpha_max
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        image = self.decode_latents(latents)
        image, has_nsfw = self.run_safety_checker(image, device, text_embeddings.dtype)

        # ── Pixel-space template injection (post-decode) ────────────────────
        if pixel_template_gamma > 0.0:
            H_img, W_img = image.shape[1], image.shape[2]
            pixel_mask = build_pixel_template_mask(H_img, W_img)
            # image is [B, H, W, 3] numpy float32 in [0,1]
            for b in range(image.shape[0]):
                image[b] = _inject_pixel_template(image[b], pixel_mask, pixel_template_gamma)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw)

        return ModifiedStableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw, init_latents=init_latents
        )
