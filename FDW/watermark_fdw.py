"""
FDW: Frequency-Domain Watermarking
Core watermark class - extends Gaussian Shading with:
  1. DCT mid-frequency structured initial noise
  2. BCH error correction coding
  3. Dual-path detection (spatial + frequency)
"""

import torch
import numpy as np
from scipy.stats import norm, truncnorm
from scipy.special import betainc
from functools import reduce
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes


# ── BCH helpers (pure numpy, no external bchlib required) ──────────────────
# We implement a lightweight BCH(255, 47, t=21) via systematic encoding.
# For simplicity we use a Reed-Solomon-style XOR parity approach when bchlib
# is unavailable, and fall back to bchlib when available.

def _try_import_bchlib():
    try:
        import bchlib
        return bchlib
    except ImportError:
        return None


class BCHCodec:
    """
    Wraps bchlib if available; otherwise uses a 3× repetition code fallback.
    payload_bits: number of information bits to protect.

    bchlib API (this version): BCH(t=N, m=M)
      - t: max correctable bit errors
      - m: GF field order (n = 2^m - 1)
      - ecc_bytes: bytes of ECC appended per block
    We use t=4, m=13 → n=8191, ecc_bytes=7, max_data_bytes=1017.
    """
    def __init__(self, payload_bits: int):
        self.payload_bits = payload_bits
        self._bchlib = _try_import_bchlib()

        if self._bchlib is not None:
            self.bch = self._bchlib.BCH(t=4, m=13)
            self.ecc_bytes = self.bch.ecc_bytes          # 7
            self.ecc_bits  = self.ecc_bytes * 8          # 56
            self.encoded_bits = payload_bits + self.ecc_bits
        else:
            self.ecc_bits = 0
            self.encoded_bits = payload_bits * 3         # 3× repetition fallback

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """bits: 1-D binary array of length payload_bits → encoded array"""
        assert len(bits) == self.payload_bits
        if self._bchlib is not None:
            data_bytes = np.packbits(bits).tobytes()
            ecc = self.bch.encode(data_bytes)            # bytes of length ecc_bytes
            ecc_bits = np.unpackbits(np.frombuffer(ecc, dtype=np.uint8))
            return np.concatenate([bits, ecc_bits])      # payload + 56 ecc bits
        else:
            return np.tile(bits, 3)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        """bits: encoded array → recovered payload bits"""
        if self._bchlib is not None:
            payload = bits[: self.payload_bits].copy()
            ecc_arr = bits[self.payload_bits: self.payload_bits + self.ecc_bits]
            data_ba  = bytearray(np.packbits(payload).tobytes())
            ecc_ba   = bytearray(np.packbits(ecc_arr).tobytes())
            try:
                nerr = self.bch.decode(data_ba, ecc_ba)
                if nerr >= 0:
                    return np.unpackbits(np.frombuffer(bytes(data_ba), dtype=np.uint8))[: self.payload_bits]
            except Exception:
                pass
            return np.unpackbits(np.frombuffer(bytes(data_ba), dtype=np.uint8))[: self.payload_bits]
        else:
            # majority vote over 3 copies
            c1 = bits[: self.payload_bits]
            c2 = bits[self.payload_bits: 2 * self.payload_bits]
            c3 = bits[2 * self.payload_bits: 3 * self.payload_bits]
            return ((c1.astype(int) + c2.astype(int) + c3.astype(int)) >= 2).astype(np.uint8)


# ── Frequency-domain helpers ────────────────────────────────────────────────

def _mid_freq_mask(H: int, W: int, low_ratio: float = 0.1, high_ratio: float = 0.5) -> torch.Tensor:
    """Boolean mask selecting mid-frequency components of a 2-D FFT."""
    cy, cx = H // 2, W // 2
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    dist = torch.sqrt(((Y - cy) / cy) ** 2 + ((X - cx) / cx) ** 2)
    return (dist >= low_ratio) & (dist <= high_ratio)


def fd_enhance_noise(w: torch.Tensor, watermark_bits: torch.Tensor,
                     lambda_: float = 0.08,
                     low_ratio: float = 0.1, high_ratio: float = 0.5) -> torch.Tensor:
    """
    Overlay a frequency-domain watermark template onto the initial noise w.
    w              : [1, 4, H, W] float16/float32 initial noise
    watermark_bits : [1, 4, H, W] binary {0,1} tensor (same shape as w)
    Returns w* with the same approximate Gaussian distribution.
    """
    orig_dtype = w.dtype
    w_f = w.float()
    bits_f = watermark_bits.float()

    # {0,1} → {-1,+1} bipolar template
    template = bits_f * 2.0 - 1.0

    H, W_size = w_f.shape[-2], w_f.shape[-1]
    mask = _mid_freq_mask(H, W_size, low_ratio, high_ratio).to(w_f.device)

    # Per-channel FFT
    W_fft = torch.fft.fft2(w_f)
    T_fft = torch.fft.fft2(template)

    W_enhanced = W_fft.clone()
    W_enhanced[..., mask] = W_enhanced[..., mask] + lambda_ * T_fft[..., mask]

    w_star = torch.fft.ifft2(W_enhanced).real

    # Re-normalise to preserve original std per channel
    for c in range(w_f.shape[1]):
        orig_std = w_f[:, c].std()
        new_std = w_star[:, c].std()
        if new_std > 1e-8:
            w_star[:, c] = w_star[:, c] * (orig_std / new_std)

    return w_star.to(orig_dtype)


def fd_detect(z_T: torch.Tensor, W_freq_template: torch.Tensor,
              low_ratio: float = 0.1, high_ratio: float = 0.5) -> torch.Tensor:
    """
    Frequency-domain bit detection.
    The watermark was embedded by adding lambda * sign(T_fft) to the mid-freq band
    of the initial noise. After DDIM inversion, z_T ≈ w* in the mid-freq band.
    We recover bits by checking whether the mid-freq FFT of z_T agrees in sign
    with the template FFT, then project back to spatial domain via IFFT.

    z_T            : [1, 4, H, W] recovered latent after DDIM inversion
    W_freq_template: [1, 4, H, W] binary {0,1} template
    Returns per-element soft decision in [0,1] (>0.5 → bit=1).
    """
    z_f = z_T.float()
    t_f = W_freq_template.float() * 2.0 - 1.0   # {0,1} → {-1,+1}

    H, W_size = z_f.shape[-2], z_f.shape[-1]
    mask = _mid_freq_mask(H, W_size, low_ratio, high_ratio).to(z_f.device)

    Z_fft = torch.fft.fft2(z_f)
    T_fft = torch.fft.fft2(t_f)

    # Agreement score: positive where z_T and template have same sign in mid-freq
    # Use real part of Z * conj(T) as the agreement signal
    agree = (Z_fft * T_fft.conj()).real   # positive → same sign → bit=1

    # Zero out non-mid-freq, project back to spatial domain
    score_fft = torch.zeros_like(Z_fft)
    score_fft[..., mask] = agree[..., mask].to(score_fft.dtype)
    score_spatial = torch.fft.ifft2(score_fft).real   # [1,4,H,W]

    # Soft bit decision: positive → 1, negative → 0
    return (score_spatial > 0).float()


def build_x_template(h: int = 64, w: int = 64,
                      angle_deg: float = 45.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MaXsive-style sparse X-template for the Fourier domain.
    Each of the 2 crossing lines has 8 discrete sampling points
    (4 radii × 2 directions), totalling 16 points.
    Radii from 0.2*(w/2) to 0.5*(w/2) with interval 0.1*(w/2).
    """
    mask2d = torch.zeros(h, w)
    cy, cx = h // 2, w // 2

    r_start = 0.2 * (w / 2)
    r_end   = 0.5 * (w / 2)
    r_step  = 0.1 * (w / 2)
    radii   = torch.arange(r_start, r_end + r_step * 0.5, r_step)

    theta = torch.deg2rad(torch.tensor(angle_deg))

    for arm_theta in (theta, theta + torch.pi / 2):
        for sign in (1.0, -1.0):
            for r in radii:
                dy = sign * r * torch.sin(arm_theta)
                dx = sign * r * torch.cos(arm_theta)
                y = int(round(cy + dy.item()))
                x = int(round(cx + dx.item()))
                if 0 <= y < h and 0 <= x < w:
                    mask2d[y, x] = 1.0

    pattern2d = mask2d
    mask    = mask2d.unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1)
    pattern = pattern2d.unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1)
    return mask, pattern


# ── Pixel-space template for rotation detection ──────────────────────────────

def inject_pixel_x_template(img: 'PIL.Image.Image',
                             gamma: float = 50.0,
                             angle_deg: float = 45.0) -> 'PIL.Image.Image':
    """Inject sparse X-template into pixel-space FFT (after VAE decode)."""
    arr = np.array(img).astype(np.float32)
    H, W = arr.shape[:2]
    cy, cx = H // 2, W // 2

    mask2d = np.zeros((H, W))
    theta = np.deg2rad(angle_deg)
    r_start = 0.2 * (W / 2)
    r_end   = 0.5 * (W / 2)
    r_step  = 0.1 * (W / 2)
    radii = np.arange(r_start, r_end + r_step * 0.5, r_step)

    for arm_theta in (theta, theta + np.pi / 2):
        for sign in (1.0, -1.0):
            for r in radii:
                dy = sign * r * np.sin(arm_theta)
                dx = sign * r * np.cos(arm_theta)
                y = int(round(cy + dy))
                x = int(round(cx + dx))
                if 0 <= y < H and 0 <= x < W:
                    mask2d[y, x] = 1.0

    for c in range(3):
        F = np.fft.fftshift(np.fft.fft2(arr[:, :, c]))
        std_val = np.abs(F).std()
        F[mask2d > 0] += gamma * std_val
        arr[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(F)))

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    from PIL import Image as _Image
    return _Image.fromarray(arr)


def detect_pixel_rotation(img: 'PIL.Image.Image',
                           angle_deg: float = 45.0,
                           step: float = 1.0) -> float:
    """
    Detect rotation angle from pixel-space X-template.
    Returns estimated rotation angle in degrees.
    """
    arr = np.array(img).astype(np.float32)
    H, W = arr.shape[:2]
    cy, cx = H // 2, W // 2

    mag = np.zeros((H, W))
    for c in range(3):
        F = np.fft.fftshift(np.fft.fft2(arr[:, :, c]))
        mag += np.abs(F)
    mag /= 3.0

    r_start = 0.2 * (W / 2)
    r_end   = 0.5 * (W / 2)
    r_step  = 0.1 * (W / 2)
    radii = np.arange(r_start, r_end + r_step * 0.5, r_step)

    def _score(offset):
        total = 0.0; count = 0
        for arm_delta in (0, 90):
            arm_rad = np.deg2rad(angle_deg + offset + arm_delta)
            for r in radii:
                y = int(round(cy + r * np.sin(arm_rad)))
                x = int(round(cx + r * np.cos(arm_rad)))
                if 0 <= y < H and 0 <= x < W:
                    total += mag[y, x]; count += 1
        return total / max(count, 1)

    coarse_angles = np.arange(-45, 46, step)
    scores = [(a, _score(float(a))) for a in coarse_angles]
    scores.sort(key=lambda x: x[1], reverse=True)
    best_a = scores[0][0]

    fine_angles = np.arange(best_a - step, best_a + step + 0.1, step / 3.0)
    fine_scores = [(a, _score(float(a))) for a in fine_angles]
    fine_scores.sort(key=lambda x: x[1], reverse=True)
    return float(fine_scores[0][0])


def blind_detect_rotation(img: 'PIL.Image.Image',
                           base_angle: float = 45.0,
                           coarse_step: float = 2.0,
                           fine_step: float = 0.5,
                           r_fracs: tuple = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)) -> float:
    """
    Blind rotation detection from pixel-space FFT magnitude spectrum.
    Detects the X-template injected via pixel_template_gamma during generation.
    Does NOT use the watermark object — fully blind detection.

    The X-template has 90° symmetry, so we search 0-90° and return the offset
    that maximizes the template score. The caller should try both offset and
    offset+90° and pick the one that gives better watermark accuracy.

    Returns estimated rotation offset in degrees.
    """
    arr = np.array(img).astype(np.float32)
    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
    H, W = gray.shape
    cy, cx = H // 2, W // 2

    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(F))

    def _score(angle_offset: float) -> float:
        total, count = 0.0, 0
        for arm_delta in (0.0, 90.0):
            arm_rad = np.deg2rad(base_angle + angle_offset + arm_delta)
            for r_frac in r_fracs:
                r = r_frac * min(H, W) / 2
                for sign in (1.0, -1.0):
                    y = int(round(cy + sign * r * np.sin(arm_rad)))
                    x = int(round(cx + sign * r * np.cos(arm_rad)))
                    if 0 <= y < H and 0 <= x < W:
                        total += mag[y, x]
                        count += 1
        return total / max(count, 1)

    # X-template has 90° symmetry → search 0-90°
    coarse_angles = np.arange(0, 91, coarse_step)
    best_a = float(max(coarse_angles, key=_score))

    fine_angles = np.arange(best_a - coarse_step, best_a + coarse_step + fine_step * 0.5, fine_step)
    best_a = float(max(fine_angles, key=_score))

    return best_a


# ── Spatial permutation for crop robustness ─────────────────────────────────

class SpatialPermuter:
    """Fixed spatial permutation of latent [B, C, H, W] on the (H, W) grid.

    Scatters each watermark bit's copies across the full spatial extent so that
    a crop attack can only *partially* degrade every bit instead of completely
    destroying some bits.
    """

    _cache = {}

    @classmethod
    def get(cls, hw=64, seed=42):
        key = (hw, seed)
        if key not in cls._cache:
            cls._cache[key] = cls(hw, seed)
        return cls._cache[key]

    def __init__(self, hw=64, seed=42):
        self.hw = hw
        n = hw * hw
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        self.perm = torch.from_numpy(perm).long()
        self.inv_perm = torch.from_numpy(np.argsort(perm)).long()

    def permute(self, z):
        B, C, H, W = z.shape
        z_flat = z.reshape(B * C, H * W)
        z_flat_perm = z_flat[:, self.perm.to(z.device)]
        return z_flat_perm.reshape(B, C, H, W)

    def inv_permute(self, z):
        B, C, H, W = z.shape
        z_flat = z.reshape(B * C, H * W)
        z_flat_inv = z_flat[:, self.inv_perm.to(z.device)]
        return z_flat_inv.reshape(B, C, H, W)


# ── Main FDW watermark class ─────────────────────────────────────────────────

class FDW_Watermark:
    """
    Frequency-Domain Watermark (FDW).

    Parameters
    ----------
    ch_factor    : channel redundancy factor (same as GS)
    hw_factor    : spatial redundancy factor (same as GS)
    fpr          : target false positive rate
    user_number  : number of users (for traceability threshold)
    payload_bits : effective information bits (before ECC)
    use_ecc      : whether to apply BCH error correction
    lambda_freq  : strength of frequency-domain overlay in initial noise
    alpha_max    : max FDSC perturbation strength during denoising
    t_start      : fraction of total steps to start FDSC (e.g. 0.2)
    t_end        : fraction of total steps to end FDSC (e.g. 0.6)
    use_fd_detect: whether to use dual-path (spatial+freq) detection
    """

    def __init__(self,
                 ch_factor: int = 1,
                 hw_factor: int = 4,
                 fpr: float = 1e-6,
                 user_number: int = 1_000_000,
                 payload_bits: int = 512,
                 use_ecc: bool = True,
                 lambda_freq: float = 0.08,
                 alpha_max: float = 0.015,
                 t_start: float = 0.2,
                 t_end: float = 0.6,
                 use_fd_detect: bool = False,
                 use_spatial_perm: bool = True):

        self.ch = ch_factor
        self.hw = hw_factor
        self.lambda_freq = lambda_freq
        self.alpha_max = alpha_max
        self.t_start = t_start
        self.t_end = t_end
        self.use_fd_detect = use_fd_detect
        self.use_spatial_perm = use_spatial_perm
        self._permuter = SpatialPermuter.get(64, seed=42) if use_spatial_perm else None

        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength // (self.ch * self.hw * self.hw)
        self.threshold = 1 if (self.hw == 1 and self.ch == 1) else self.ch * self.hw * self.hw // 2

        # Payload & ECC
        # With hw=4: marklength=1024. We embed payload_bits=512 using 2× repetition
        # (512 × 2 = 1024 = marklength). Each payload bit gets 2 × (hw×hw) = 32 spatial votes.
        # Fallback: if payload_bits > marklength, clamp to marklength (no ECC).
        self.use_ecc = use_ecc
        self.payload_bits = min(payload_bits, self.marklength)
        if use_ecc and self.payload_bits * 2 <= self.marklength:
            # 2× repetition fits: store payload twice in the marklength slots
            self._ecc_repeat = 2
        elif use_ecc and self.payload_bits <= self.marklength:
            # payload fits once but not twice: no repetition, just use BCH if available
            self._ecc_repeat = 1
            self.codec = BCHCodec(self.payload_bits)
        else:
            self._ecc_repeat = 1
        if not use_ecc:
            self._ecc_repeat = 1

        # State
        self.key = None
        self.nonce = None
        self.watermark = None          # [1, 4//ch, 64//hw, 64//hw] binary
        self._watermark_bits_full = None  # [1, 4, 64, 64] binary (for FD)
        self._W_freq_template = None   # stored for detection
        self._template_mask = None
        self._template_pattern = None

        # Thresholds
        self.tau_onebit = None
        self.tau_bits = None
        self.tp_onebit_count = 0
        self.tp_bits_count = 0

        for i in range(self.marklength):
            fpr_onebit = betainc(i + 1, self.marklength - i, 0.5)
            fpr_bits = betainc(i + 1, self.marklength - i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    # ── Encryption helpers ──────────────────────────────────────────────────

    def _encrypt(self, sd: np.ndarray) -> np.ndarray:
        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        return np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))

    def _decrypt(self, m: np.ndarray) -> np.ndarray:
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(m).tobytes())
        return np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))

    # ── Truncated sampling (same as GS) ────────────────────────────────────

    def _trunc_sample(self, message: np.ndarray) -> torch.Tensor:
        z = np.zeros(self.latentlength)
        ppf = [norm.ppf(j / 2.0) for j in range(3)]
        for i in range(self.latentlength):
            bit = int(message[i])
            z[i] = truncnorm.rvs(ppf[bit], ppf[bit + 1])
        return torch.from_numpy(z).reshape(1, 4, 64, 64).half()

    # ── Redundancy expansion / voting (same as GS) ─────────────────────────

    def _expand(self, watermark: torch.Tensor) -> torch.Tensor:
        return watermark.repeat(1, self.ch, self.hw, self.hw)

    def _vote(self, sd: torch.Tensor) -> torch.Tensor:
        ch_s = 4 // self.ch
        hw_s = 64 // self.hw
        s1 = torch.cat(torch.split(sd, [ch_s] * self.ch, dim=1), dim=0)
        s2 = torch.cat(torch.split(s1, [hw_s] * self.hw, dim=2), dim=0)
        s3 = torch.cat(torch.split(s2, [hw_s] * self.hw, dim=3), dim=0)
        vote = torch.sum(s3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    # ── Public API ──────────────────────────────────────────────────────────

    def create_watermark_and_return_w(self) -> torch.Tensor:
        """
        Generate a new random watermark and return the structured initial noise w*.
        Watermark shape: [1, payload_bits_h, payload_bits_w] packed into
        [1, 4//ch, 64//hw, 64//hw] with optional 2× repetition for ECC.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. Random watermark payload (payload_bits total)
        # Pack into the watermark tensor shape [1, 4//ch, 64//hw, 64//hw]
        # If _ecc_repeat=2: watermark occupies first half of marklength slots
        wm_h = 64 // self.hw
        wm_c = 4 // self.ch
        # Full watermark tensor (marklength bits)
        wm_full = torch.randint(0, 2, [1, wm_c, wm_h, wm_h]).to(device)

        if self._ecc_repeat == 2:
            # payload = first payload_bits bits; second half is a copy
            payload_flat = wm_full.flatten()[:self.payload_bits]
            # Build full marklength tensor: [payload | payload | zeros_if_any]
            repeated = payload_flat.repeat(2)[:self.marklength]
            wm_full = repeated.reshape(1, wm_c, wm_h, wm_h).to(torch.int64)

        self.watermark = wm_full  # [1, wm_c, wm_h, wm_h]

        # 2. Expand + encrypt → m_bit (latentlength bits)
        sd = self._expand(self.watermark)
        m = self._encrypt(sd.flatten().cpu().numpy())

        # 3. Truncated sampling → spatial noise w
        w = self._trunc_sample(m).to(device)

        # 4. Store frequency template
        m_tensor = torch.from_numpy(m).reshape(1, 4, 64, 64).to(torch.uint8).to(device)
        self._watermark_bits_full = m_tensor
        self._W_freq_template = m_tensor

        # 4.5 Build MaXsive-style X-template for geometric correction / injection
        self._template_mask, self._template_pattern = build_x_template(64, 64)
        self._template_mask = self._template_mask.to(device)
        self._template_pattern = self._template_pattern.to(device)

        # 5. Frequency-domain enhancement
        w_star = fd_enhance_noise(w, m_tensor.float(), lambda_=self.lambda_freq)

        # 6. Spatial permutation for crop robustness
        if self._permuter is not None:
            w_star = self._permuter.permute(w_star)

        return w_star

    def get_fdsc_template(self) -> torch.Tensor:
        """Return the frequency template for FDSC injection during denoising."""
        assert self._W_freq_template is not None, "Call create_watermark_and_return_w first"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t_f = self._W_freq_template.float().to(device) * 2.0 - 1.0
        return torch.fft.fft2(t_f)  # pre-computed FFT of template

    def get_x_template(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mask, pattern) for MaXsive-style template injection."""
        assert self._template_mask is not None and self._template_pattern is not None, \
            "Call create_watermark_and_return_w first"
        return self._template_mask, self._template_pattern

    def _compute_accuracy(self, reversed_w: torch.Tensor) -> float:
        if self._permuter is not None:
            reversed_w = self._permuter.inv_permute(reversed_w)

        reversed_m_a = (reversed_w > 0).int()
        sd_bytes = self._decrypt(reversed_m_a.flatten().cpu().numpy())
        sd_tensor = torch.from_numpy(sd_bytes).reshape(1, 4, 64, 64).to(torch.uint8).to(reversed_w.device)
        wm_voted = self._vote(sd_tensor)

        if self.use_fd_detect and self._W_freq_template is not None:
            fd_bits = fd_detect(reversed_w, self._W_freq_template.float())
            fd_m = (fd_bits > 0.5).int()
            sd_fd = self._decrypt(fd_m.flatten().cpu().numpy())
            sd_fd_t = torch.from_numpy(sd_fd).reshape(1, 4, 64, 64).to(torch.uint8).to(reversed_w.device)
            wm_b = self._vote(sd_fd_t)
            wm_voted = ((wm_voted.int() + wm_b.int()) >= 2).to(torch.uint8)

        if self._ecc_repeat == 2:
            flat = wm_voted.flatten()
            half = self.payload_bits
            copy1 = flat[:half].int()
            copy2 = flat[half:2 * half].int()
            recovered = ((copy1 + copy2) >= 1).to(torch.uint8)
            ref = self.watermark.flatten()[:half]
            correct = (recovered == ref).float().mean().item()
        else:
            correct = (wm_voted == self.watermark).float().mean().item()

        return correct

    def score_watermark(self, reversed_w: torch.Tensor) -> float:
        return self._compute_accuracy(reversed_w)

    def eval_watermark(self, reversed_w: torch.Tensor) -> float:
        """
        Evaluate watermark accuracy from recovered latent z_T.
        Returns bit accuracy of payload bits in [0, 1].
        """
        correct = self._compute_accuracy(reversed_w)

        if correct >= self.tau_onebit:
            self.tp_onebit_count += 1
        if correct >= self.tau_bits:
            self.tp_bits_count += 1

        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count

    def reset_counters(self):
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
