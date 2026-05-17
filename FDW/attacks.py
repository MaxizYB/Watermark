"""
Attack library for watermark robustness evaluation.
Covers all attack categories from the PPT:
  - Clean (no attack)
  - Geometric: rotate, scale, crop, translate, flip
  - Photometric: brightness, contrast, saturation, hue
  - Degradation: JPEG, Gaussian blur, median blur, Gaussian noise, S&P noise, resize
  - Adversarial: FGSM-style pixel perturbation
  - Regeneration: VAE re-encode/decode (requires pipe)
  - Stirmark: combined geometric distortions
"""

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
import random
import io
from typing import Optional, Callable


# ── Type alias ────────────────────────────────────────────────────────────────
AttackFn = Callable[[Image.Image], Image.Image]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_pil(img) -> Image.Image:
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype(np.uint8))
    return img


def _to_np(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# 1. Clean
# ══════════════════════════════════════════════════════════════════════════════

def attack_clean(img: Image.Image) -> Image.Image:
    return img


# ══════════════════════════════════════════════════════════════════════════════
# 2. Geometric attacks
# ══════════════════════════════════════════════════════════════════════════════

def attack_rotate(img: Image.Image, angle: float = 45.0) -> Image.Image:
    return img.rotate(angle, expand=False, fillcolor=(0, 0, 0))


def attack_scale(img: Image.Image, ratio: float = 0.75) -> Image.Image:
    w, h = img.size
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    # Keep image content and pad back to original size (MaXsive/Stirmark-style)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
    return canvas


def estimate_rotation_scale_from_x_template(img: Image.Image) -> tuple[float, float]:
    """
    Estimate rotation angle and rough scale factor from an X-shaped template in the
    Fourier magnitude domain, following the spirit of MaXsive Sec. 4.3.1.
    Returns (angle_deg, scale_ratio).

    IMPORTANT: this only works if the image actually contains a visible/stable X-template
    in its Fourier magnitude. On ordinary images without such a template, estimates are noisy.
    """
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    F = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.log1p(np.abs(F))
    H, W = mag.shape
    cy, cx = H // 2, W // 2

    def line_samples(theta_deg: float):
        theta = np.deg2rad(theta_deg)
        vals = []
        coords = []
        for sign in (-1, 1):
            phi = theta + sign * np.pi / 4
            for r in np.linspace(0.18, 0.48, 64):
                y = int(round(cy + r * H * np.sin(phi)))
                x = int(round(cx + r * W * np.cos(phi)))
                if 0 <= y < H and 0 <= x < W:
                    vals.append(mag[y, x])
                    coords.append((r, y, x))
        return vals, coords

    # Coarse-to-fine search for dominant X angle
    coarse = np.arange(-45, 46, 3)
    coarse_best = max(coarse, key=lambda th: np.mean(line_samples(th)[0]) if line_samples(th)[0] else 0.0)
    fine = np.arange(coarse_best - 3, coarse_best + 3.1, 0.5)
    best_theta = max(fine, key=lambda th: np.mean(line_samples(th)[0]) if line_samples(th)[0] else 0.0)

    # Estimate radius of strongest responses along the best diagonal pair
    vals, coords = line_samples(best_theta)
    radii = []
    if vals:
        vals = np.asarray(vals)
        coords = np.asarray(coords, dtype=object)
        # take top-16 strongest points, average their radii
        topk = np.argsort(vals)[-16:]
        radii = [float(coords[i][0]) for i in topk]
    mean_r = float(np.mean(radii)) if radii else 0.30

    # Reference radius ~0.30 in our default template; inverse relation to scale.
    scale_ratio = max(0.7, min(1.3, 0.30 / max(mean_r, 1e-6)))
    return float(best_theta), float(scale_ratio)


def correct_rotation_scale(img: Image.Image, angle_deg: float, scale_ratio: float) -> Image.Image:
    """Undo estimated rotation and scale while keeping output size fixed."""
    corrected = img.rotate(-angle_deg, expand=False, fillcolor=(0, 0, 0))
    if abs(scale_ratio - 1.0) > 1e-3:
        w, h = corrected.size
        new_w, new_h = int(round(w * scale_ratio)), int(round(h * scale_ratio))
        scaled = corrected.resize((max(1, new_w), max(1, new_h)), Image.BICUBIC)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(scaled, ((w - scaled.size[0]) // 2, (h - scaled.size[1]) // 2))
        corrected = canvas
    return corrected


# ══════════════════════════════════════════════════════════════════════════════
# Latent-space template detection & correction (MaXsive-style)
# ══════════════════════════════════════════════════════════════════════════════

def detect_template_angle_in_zT(z_T: torch.Tensor,
                                 base_angle: float = 45.0,
                                 r_min: float = 0.20,
                                 r_max: float = 0.50,
                                 n_points: int = 8) -> float:
    z_avg = z_T.float().mean(dim=1, keepdim=True)
    Z = torch.fft.fftshift(torch.fft.fft2(z_avg), dim=(-2, -1))
    mag = Z.abs().squeeze()
    H_s, W_s = mag.shape
    cy, cx = H_s // 2, W_s // 2

    def _score(angle_offset: float):
        total = 0.0
        count = 0
        for arm_delta in (0, 90):
            arm_rad = np.deg2rad(base_angle + angle_offset + arm_delta)
            for r in np.linspace(r_min, r_max, n_points):
                y = int(round(cy + r * H_s * np.sin(arm_rad)))
                x = int(round(cx + r * W_s * np.cos(arm_rad)))
                if 0 <= y < H_s and 0 <= x < W_s:
                    total += mag[y, x].item()
                    count += 1
        return total / max(count, 1)

    best_a, best_s = 0.0, _score(0.0)
    for a in range(-45, 46, 3):
        s = _score(float(a))
        if s > best_s:
            best_s, best_a = s, float(a)
    for a in np.arange(best_a - 3, best_a + 3.1, 0.5):
        s = _score(float(a))
        if s > best_s:
            best_s, best_a = s, float(a)
    return best_a


def _bilinear_sample(mag: np.ndarray, y: float, x: float) -> float:
    H, W = mag.shape
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    fx = x - x0; fy = y - y0
    def _safe(yy, xx):
        return mag[yy % H, xx % W]
    return (_safe(y0, x0) * (1-fx) * (1-fy) + _safe(y0, x0+1) * fx * (1-fy) +
            _safe(y0+1, x0) * (1-fx) * fy + _safe(y0+1, x0+1) * fx * fy)


def detect_rotation_ring(z_T: torch.Tensor) -> float:
    z_avg = z_T.float().mean(dim=1, keepdim=True)
    Z = torch.fft.fftshift(torch.fft.fft2(z_avg), dim=(-2, -1))
    mag = Z.abs().squeeze().cpu().numpy()
    H, W = mag.shape
    cy, cx = H / 2.0, W / 2.0
    n_angles = 720
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    profile = np.zeros(n_angles)
    radii = np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]) * (W / 2)
    for i, a_deg in enumerate(angles):
        a_rad = np.deg2rad(a_deg)
        sa, ca = np.sin(a_rad), np.cos(a_rad)
        total = 0.0; count = 0
        for r in radii:
            y = cy + r * sa; x = cx + r * ca
            y0 = int(np.floor(y)); x0 = int(np.floor(x))
            if 0 <= y0 < H-1 and 0 <= x0 < W-1:
                fy = y - y0; fx = x - x0
                total += (mag[y0,x0]*(1-fx)*(1-fy) + mag[y0,x0+1]*fx*(1-fy) +
                          mag[y0+1,x0]*(1-fx)*fy + mag[y0+1,x0+1]*fx*fy)
                count += 1
        profile[i] = total / max(count, 1)
    template = np.zeros(n_angles)
    for base_a in [45, 135]:
        idx = int(round(base_a / 360 * n_angles)) % n_angles
        w = 3
        for d in range(-w, w+1):
            template[(idx+d) % n_angles] = max(template[(idx+d) % n_angles],
                                                 np.exp(-0.5*(d/w*2)**2))
    corr = np.real(np.fft.ifft(np.fft.fft(profile) * np.conj(np.fft.fft(template))))
    best_idx = int(np.argmax(corr))
    detected = angles[best_idx]
    if detected > 180:
        detected -= 360
    return float(detected)


def detect_and_correct_rotation(z_T: torch.Tensor,
                                 attacked_img: Image.Image,
                                 img_size: int = 512,
                                 score_fn=None) -> tuple:
    detected = detect_rotation_ring(z_T)
    corr_angle = (-detected) % 90
    if corr_angle > 45:
        corr_angle -= 90

    if score_fn is not None:
        candidates = [-detected, -detected + 90, -detected - 90, -detected + 180]
        candidates = list(set(candidates))
        best_acc = -1.0
        best_corr = corr_angle
        best_img = None
        for cand in candidates:
            unrot = attacked_img.rotate(-cand, resample=Image.BICUBIC,
                                         expand=True, fillcolor=(255, 255, 255))
            uw, uh = unrot.size
            left = (uw - img_size) // 2
            top = (uh - img_size) // 2
            crop = unrot.crop((left, top, left + img_size, top + img_size))
            acc = score_fn(crop)
            if acc > best_acc:
                best_acc = acc
                best_corr = cand
                best_img = crop
        return best_img, detected, best_corr

    unrotated = attacked_img.rotate(-corr_angle, resample=Image.BICUBIC,
                                     expand=True, fillcolor=(255, 255, 255))
    uw, uh = unrotated.size
    left = (uw - img_size) // 2
    top = (uh - img_size) // 2
    corrected = unrotated.crop((left, top, left + img_size, top + img_size))
    return corrected, detected, corr_angle


def detect_black_border(img: Image.Image, threshold: int = 15) -> tuple:
    """
    Detect black border region in image (from scale/pad attacks).
    Returns (left, top, right, bottom) bounding box of non-black content.
    If no black border detected, returns the full image bounds.
    """
    arr = np.array(img.convert('L'))
    mask = arr > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows):
        return 0, 0, arr.shape[1], arr.shape[0]
    row_start, row_end = np.argmax(rows), arr.shape[0] - np.argmax(rows[::-1])
    col_start, col_end = np.argmax(cols), arr.shape[1] - np.argmax(cols[::-1])
    return int(col_start), int(row_start), int(col_end), int(row_end)


def detect_and_correct_scale(img: Image.Image, target_size: int = 512,
                              black_threshold: int = 15) -> tuple:
    """
    Detect and correct scale attack: crop symmetric black border, resize back to target.
    Only triggers when black border is centered and symmetric (scale attack),
    NOT when it's off-center (crop attack).
    Returns (corrected_img, detected_ratio, was_corrected).
    """
    W, H = img.size
    left, top, right, bottom = detect_black_border(img, black_threshold)
    content_w = right - left
    content_h = bottom - top

    if content_w <= 0 or content_h <= 0:
        return img, 1.0, False

    border_top = top
    border_bottom = H - bottom
    border_left = left
    border_right = W - right

    min_border = min(border_top, border_bottom, border_left, border_right)
    has_border = min_border > H * 0.03

    if not has_border:
        return img, 1.0, False

    symmetry_h = abs(border_top - border_bottom) / max(max(border_top, border_bottom), 1)
    symmetry_w = abs(border_left - border_right) / max(max(border_left, border_right), 1)
    is_centered = symmetry_h < 0.35 and symmetry_w < 0.35

    if not is_centered:
        return img, 1.0, False

    cropped = img.crop((left, top, right, bottom))
    corrected = cropped.resize((target_size, target_size), Image.BICUBIC)
    ratio = content_w / W
    return corrected, ratio, True


def content_crop_resize(img: Image.Image, target_size: int = 512,
                         black_threshold: int = 15) -> tuple:
    """
    Aggressive content cropping: remove any black borders and resize.
    Does NOT check for symmetry — caller must decide when to use this.
    Returns (resized_img, was_cropped).
    """
    W, H = img.size
    left, top, right, bottom = detect_black_border(img, black_threshold)
    content_w = right - left
    content_h = bottom - top

    if content_w <= 0 or content_h <= 0:
        return img, False

    cropped = img.crop((left, top, right, bottom))
    corrected = cropped.resize((target_size, target_size), Image.BICUBIC)
    return corrected, True


def has_asymmetric_borders(img: Image.Image, img_size: int = 512,
                            min_content_ratio: float = 0.5) -> bool:
    """
    Check if image has significant black borders that are NOT symmetric.
    Returns True only if borders exist but are off-center (translation-like).
    """
    W, H = img.size
    left, top, right, bottom = detect_black_border(img)
    content_w = right - left
    content_h = bottom - top
    if content_w <= 0 or content_h <= 0:
        return False
    min_border = min(left, top, W - right, H - bottom)
    if min_border <= H * 0.03:
        return False
    content_ratio = (content_w * content_h) / (W * H)
    if content_ratio < min_content_ratio:
        return False
    border_top = top
    border_bottom = H - bottom
    border_left = left
    border_right = W - right
    symmetry_h = abs(border_top - border_bottom) / max(max(border_top, border_bottom), 1)
    symmetry_w = abs(border_left - border_right) / max(max(border_left, border_right), 1)
    is_symmetric = symmetry_h < 0.35 and symmetry_w < 0.35
    return not is_symmetric


def detect_and_correct_geom(z_T: torch.Tensor,
                             attacked_img: Image.Image,
                             img_size: int = 512,
                             score_fn=None) -> tuple:
    """
    Unified geometric correction.
    1. Symmetric borders → scale correction → rotation
    2. No borders → rotation only
    If score_fn is provided, uses 4-way rotation search to resolve 90° ambiguity.
    score_fn: callable(PIL.Image) -> float, returns watermark accuracy.
    """
    corrected, scale_ratio, scale_corrected = detect_and_correct_scale(attacked_img, img_size)

    detected_angle = 0.0
    rot_corrected = False
    base_img = corrected if scale_corrected else attacked_img
    corrected, detected_angle, _ = detect_and_correct_rotation(
        z_T, base_img, img_size, score_fn=score_fn)
    rot_corrected = abs(detected_angle) > 0.5

    return corrected, detected_angle, scale_corrected, rot_corrected


def correct_zT_rotation(z_T: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    Undo estimated rotation in latent space (on z_T' after DDIM inversion).
    z_T: [1, 4, H, W] tensor.  Positive angle_deg = CCW rotation to undo.
    Uses NEAREST interpolation to preserve sign for watermark detection.
    """
    if abs(angle_deg) < 0.1:
        return z_T
    from torchvision.transforms.functional import rotate as torch_rotate
    from torchvision.transforms import InterpolationMode
    return torch_rotate(z_T.float(), -angle_deg, fill=0,
                        interpolation=InterpolationMode.NEAREST).to(z_T.dtype)


def attack_crop(img: Image.Image, ratio: float = 0.6) -> Image.Image:
    """Random crop keeping ratio of area, pad zeros elsewhere."""
    arr = np.array(img)
    H, W = arr.shape[:2]
    new_H, new_W = int(H * ratio), int(W * ratio)
    y0 = random.randint(0, H - new_H)
    x0 = random.randint(0, W - new_W)
    out = np.zeros_like(arr)
    out[y0:y0 + new_H, x0:x0 + new_W] = arr[y0:y0 + new_H, x0:x0 + new_W]
    return Image.fromarray(out)


def attack_translate(img: Image.Image, tx: int = 30, ty: int = 30) -> Image.Image:
    """Translate image by (tx, ty) pixels, filling vacated area with zeros."""
    arr = np.array(img)
    H, W = arr.shape[:2]
    out = np.zeros_like(arr)
    # Source region in original image
    src_x0, src_x1 = max(0, -tx), min(W, W - tx)
    src_y0, src_y1 = max(0, -ty), min(H, H - ty)
    # Destination region in output image
    dst_x0, dst_x1 = max(0, tx), min(W, W + tx)
    dst_y0, dst_y1 = max(0, ty), min(H, H + ty)
    # Clip to valid sizes (both regions must have same shape)
    copy_h = min(src_y1 - src_y0, dst_y1 - dst_y0)
    copy_w = min(src_x1 - src_x0, dst_x1 - dst_x0)
    if copy_h > 0 and copy_w > 0:
        out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = \
            arr[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
    return Image.fromarray(out)


def attack_flip_h(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def attack_flip_v(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_TOP_BOTTOM)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Photometric attacks
# ══════════════════════════════════════════════════════════════════════════════

def attack_brightness(img: Image.Image, factor: float = 2.0) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def attack_contrast(img: Image.Image, factor: float = 2.0) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def attack_saturation(img: Image.Image, factor: float = 2.0) -> Image.Image:
    return ImageEnhance.Color(img).enhance(factor)


def attack_color_jitter(img: Image.Image,
                        brightness: float = 0.5,
                        contrast: float = 0.5,
                        saturation: float = 0.5,
                        hue: float = 0.1) -> Image.Image:
    t = transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                saturation=saturation, hue=hue)
    return t(img)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Degradation attacks
# ══════════════════════════════════════════════════════════════════════════════

def attack_jpeg(img: Image.Image, quality: int = 50) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def attack_gaussian_blur(img: Image.Image, radius: float = 3.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def attack_median_blur(img: Image.Image, size: int = 7) -> Image.Image:
    return img.filter(ImageFilter.MedianFilter(size=size))


def attack_gaussian_noise(img: Image.Image, std: float = 0.05) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std * 255, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def attack_sp_noise(img: Image.Image, prob: float = 0.05) -> Image.Image:
    arr = np.array(img).copy()
    rnd = np.random.rand(*arr.shape[:2])
    arr[rnd < prob / 2] = 0
    arr[rnd > 1 - prob / 2] = 255
    return Image.fromarray(arr)


def attack_resize(img: Image.Image, ratio: float = 0.25) -> Image.Image:
    w, h = img.size
    small = img.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def attack_pixelate(img: Image.Image, block_size: int = 16) -> Image.Image:
    w, h = img.size
    small = img.resize((w // block_size, h // block_size), Image.BOX)
    return small.resize((w, h), Image.NEAREST)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Adversarial attack (FGSM-style, no model needed)
# ══════════════════════════════════════════════════════════════════════════════

def attack_adversarial_noise(img: Image.Image, epsilon: float = 8.0) -> Image.Image:
    """
    Add structured adversarial-like noise (random sign pattern scaled by epsilon).
    True FGSM requires a model; this approximates the pixel-level effect.
    """
    arr = np.array(img).astype(np.float32)
    sign_noise = np.sign(np.random.randn(*arr.shape)) * epsilon
    return Image.fromarray(np.clip(arr + sign_noise, 0, 255).astype(np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
# 6. Regeneration attack (requires diffusion pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def attack_regeneration(img: Image.Image, pipe, strength: float = 0.3,
                        num_steps: int = 20, prompt: str = "") -> Image.Image:
    """
    Re-generate the image through the diffusion model (img2img style).
    strength: how much noise to add before re-denoising (0=no change, 1=full noise)
    Requires a diffusers StableDiffusionImg2ImgPipeline or compatible pipe.
    """
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        import torch
        device = next(pipe.parameters()).device if hasattr(pipe, 'parameters') else 'cuda'
        # Use the pipe's VAE to encode/decode
        t = transforms.ToTensor()(img).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            latent = pipe.vae.encode(t.half()).latent_dist.mode() * 0.18215
            # Add noise proportional to strength
            noise = torch.randn_like(latent) * strength
            latent_noisy = latent + noise
            decoded = pipe.vae.decode(latent_noisy / 0.18215).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded_np = decoded.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return Image.fromarray((decoded_np * 255).astype(np.uint8))
    except Exception as e:
        print(f"[Regeneration attack] Failed: {e}. Returning original.")
        return img


# ══════════════════════════════════════════════════════════════════════════════
# 7. Stirmark-style combined attacks
# ══════════════════════════════════════════════════════════════════════════════

def attack_stirmark_rst(img: Image.Image) -> Image.Image:
    """Rotation + Scale + Translation combined."""
    img = attack_rotate(img, angle=random.uniform(-15, 15))
    img = attack_scale(img, ratio=random.uniform(0.8, 1.2))
    img = attack_translate(img, tx=random.randint(-20, 20), ty=random.randint(-20, 20))
    return img


def attack_stirmark_all(img: Image.Image) -> Image.Image:
    """Full Stirmark: geometric + photometric + degradation."""
    img = attack_stirmark_rst(img)
    img = attack_jpeg(img, quality=random.randint(50, 85))
    img = attack_gaussian_noise(img, std=random.uniform(0.01, 0.03))
    img = attack_brightness(img, factor=random.uniform(0.8, 1.2))
    return img


# ══════════════════════════════════════════════════════════════════════════════
# Attack registry
# ══════════════════════════════════════════════════════════════════════════════

ATTACK_REGISTRY = {
    # Clean
    "clean": attack_clean,

    # Geometric
    "rotate_15":    lambda img: attack_rotate(img, 15),
    "rotate_45":    lambda img: attack_rotate(img, 45),
    "scale_075":    lambda img: attack_scale(img, 0.75),
    "scale_050":    lambda img: attack_scale(img, 0.50),
    "crop_080":     lambda img: attack_crop(img, 0.80),
    "crop_060":     lambda img: attack_crop(img, 0.60),
    "translate":    lambda img: attack_translate(img, 30, 30),
    "flip_h":       attack_flip_h,
    "flip_v":       attack_flip_v,

    # Photometric
    "brightness_2": lambda img: attack_brightness(img, 2.0),
    "brightness_6": lambda img: attack_brightness(img, 6.0),
    "contrast_2":   lambda img: attack_contrast(img, 2.0),
    "saturation_2": lambda img: attack_saturation(img, 2.0),
    "color_jitter": attack_color_jitter,

    # Degradation
    "jpeg_75":      lambda img: attack_jpeg(img, 75),
    "jpeg_50":      lambda img: attack_jpeg(img, 50),
    "jpeg_25":      lambda img: attack_jpeg(img, 25),
    "gauss_blur_2": lambda img: attack_gaussian_blur(img, 2.0),
    "gauss_blur_4": lambda img: attack_gaussian_blur(img, 4.0),
    "median_blur_7":lambda img: attack_median_blur(img, 7),
    "gauss_noise_005": lambda img: attack_gaussian_noise(img, 0.05),
    "gauss_noise_010": lambda img: attack_gaussian_noise(img, 0.10),
    "sp_noise_005": lambda img: attack_sp_noise(img, 0.05),
    "resize_050":   lambda img: attack_resize(img, 0.50),
    "resize_025":   lambda img: attack_resize(img, 0.25),
    "pixelate_16":  lambda img: attack_pixelate(img, 16),

    # Adversarial
    "adversarial_8":  lambda img: attack_adversarial_noise(img, 8.0),
    "adversarial_16": lambda img: attack_adversarial_noise(img, 16.0),

    # Stirmark
    "stirmark_rst": attack_stirmark_rst,
    "stirmark_all": attack_stirmark_all,
}

# Grouped for convenience
ATTACK_GROUPS = {
    "clean":        ["clean"],
    "geometric":    ["rotate_15", "rotate_45", "scale_075", "scale_050",
                     "crop_080", "crop_060", "translate", "flip_h"],
    "photometric":  ["brightness_2", "brightness_6", "contrast_2",
                     "saturation_2", "color_jitter"],
    "degradation":  ["jpeg_75", "jpeg_50", "jpeg_25",
                     "gauss_blur_2", "gauss_blur_4", "median_blur_7",
                     "gauss_noise_005", "gauss_noise_010",
                     "sp_noise_005", "resize_050", "resize_025"],
    "adversarial":  ["adversarial_8", "adversarial_16"],
    "stirmark":     ["stirmark_rst", "stirmark_all"],
}


def get_attack(name: str, pipe=None) -> AttackFn:
    """Return attack function by name. Pass pipe for regeneration attack.
    Supports dynamic parameterized names: rotate_X, scale_XXX, crop_XXX, jpeg_X.
    """
    if name == "regeneration":
        if pipe is None:
            raise ValueError("Regeneration attack requires a diffusion pipeline.")
        return lambda img: attack_regeneration(img, pipe)
    if name in ATTACK_REGISTRY:
        return ATTACK_REGISTRY[name]
    import re
    m = re.match(r'rotate_([-\d.]+)$', name)
    if m:
        a = float(m.group(1))
        return lambda img, _a=a: attack_rotate(img, _a)
    m = re.match(r'scale_(\d+)$', name)
    if m:
        r = int(m.group(1)) / 100.0
        return lambda img, _r=r: attack_scale(img, _r)
    m = re.match(r'crop_(\d+)$', name)
    if m:
        r = int(m.group(1)) / 100.0
        return lambda img, _r=r: attack_crop(img, _r)
    m = re.match(r'jpeg_(\d+)$', name)
    if m:
        q = int(m.group(1))
        return lambda img, _q=q: attack_jpeg(img, _q)
    m = re.match(r'resize_(\d+)$', name)
    if m:
        r = int(m.group(1)) / 100.0
        return lambda img, _r=r: attack_resize(img, _r)
    m = re.match(r'gauss_noise_(\d+)$', name)
    if m:
        s = int(m.group(1)) / 100.0
        return lambda img, _s=s: attack_gaussian_noise(img, _s)
    m = re.match(r'gauss_blur_([\d.]+)$', name)
    if m:
        r = float(m.group(1))
        return lambda img, _r=r: attack_gaussian_blur(img, _r)
    m = re.match(r'brightness_([\d.]+)$', name)
    if m:
        f = float(m.group(1))
        return lambda img, _f=f: attack_brightness(img, _f)
    raise KeyError(f"Unknown attack '{name}'. Available: {list(ATTACK_REGISTRY.keys())}")


def list_attacks() -> list:
    return list(ATTACK_REGISTRY.keys()) + ["regeneration"]
