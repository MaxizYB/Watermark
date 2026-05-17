"""
DwtDct baseline watermark: classic DWT + DCT + QIM watermarking.

Post-processing watermark applied to generated images (not latent space).
256 bits capacity, matches Gaussian Shading capacity for fair comparison.

Flow:
  embed: PIL Image → Y channel → Haar DWT → block DCT on HL subband → QIM embed → reconstruct
  extract: attacked image → same transforms → QIM decode → recovered bits
"""

import numpy as np
from PIL import Image


def _haar_dwt2d(arr: np.ndarray) -> tuple:
    r, c = arr.shape
    s = np.sqrt(2.0)
    temp = np.zeros_like(arr)
    temp[:r // 2, :] = (arr[0::2, :] + arr[1::2, :]) / s
    temp[r // 2:, :] = (arr[0::2, :] - arr[1::2, :]) / s
    LL = (temp[:r // 2, 0::2] + temp[:r // 2, 1::2]) / s
    LH = (temp[:r // 2, 0::2] - temp[:r // 2, 1::2]) / s
    HL = (temp[r // 2:, 0::2] + temp[r // 2:, 1::2]) / s
    HH = (temp[r // 2:, 0::2] - temp[r // 2:, 1::2]) / s
    return LL, LH, HL, HH


def _haar_idwt2d(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray) -> np.ndarray:
    r2, c2 = LL.shape
    s = np.sqrt(2.0)
    temp_lo = np.zeros((r2, c2 * 2))
    temp_lo[:, 0::2] = (LL + LH) / s
    temp_lo[:, 1::2] = (LL - LH) / s
    temp_hi = np.zeros((r2, c2 * 2))
    temp_hi[:, 0::2] = (HL + HH) / s
    temp_hi[:, 1::2] = (HL - HH) / s
    result = np.zeros((r2 * 2, c2 * 2))
    result[0::2, :] = (temp_lo + temp_hi) / s
    result[1::2, :] = (temp_lo - temp_hi) / s
    return result


def _dct2d(block: np.ndarray) -> np.ndarray:
    N = block.shape[0]
    ns = np.arange(N)
    C = np.zeros((N, N))
    C[0, :] = 1.0 / np.sqrt(N)
    for k in range(1, N):
        C[k, :] = np.sqrt(2.0 / N) * np.cos(np.pi * (ns + 0.5) * k / N)
    return C @ block @ C.T


def _idct2d(block: np.ndarray) -> np.ndarray:
    N = block.shape[0]
    ns = np.arange(N)
    C = np.zeros((N, N))
    C[0, :] = 1.0 / np.sqrt(N)
    for k in range(1, N):
        C[k, :] = np.sqrt(2.0 / N) * np.cos(np.pi * (ns + 0.5) * k / N)
    return C.T @ block @ C


class DwtDctWatermark:
    def __init__(self, payload_bits: int = 256, key: int = 42, delta: float = 80.0):
        self.payload_bits = payload_bits
        self.key = key
        self.delta = delta
        self.rng = np.random.RandomState(key)
        self.wm_bits = self.rng.randint(0, 2, payload_bits).astype(np.int32)
        self._build_pos_map()

    def _build_pos_map(self):
        rng = np.random.RandomState(self.key + 100)
        bsz = 16
        n_blocks = 16
        freq_range = list(range(2, bsz - 2))
        positions = []
        for i in range(n_blocks):
            for j in range(n_blocks):
                fi = freq_range[(i * 7 + j * 3) % len(freq_range)]
                fj = freq_range[(i * 5 + j * 11) % len(freq_range)]
                positions.append((i, j, fi, fj))
        rng.shuffle(positions)
        self.positions = positions[: self.payload_bits]

    def embed(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float64)
        Y = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        LL, LH, HL, HH = _haar_dwt2d(Y)
        bsz = 16
        n_blocks = HL.shape[0] // bsz
        HL_mod = HL.copy()
        for k in range(self.payload_bits):
            bi, bj, fi, fj = self.positions[k]
            r0, c0 = bi * bsz, bj * bsz
            block = HL_mod[r0 : r0 + bsz, c0 : c0 + bsz].copy()
            dct_block = _dct2d(block)
            c_val = dct_block[fi, fj]
            if self.wm_bits[k] == 0:
                dct_block[fi, fj] = np.round(c_val / self.delta) * self.delta
            else:
                dct_block[fi, fj] = np.round(c_val / self.delta) * self.delta + self.delta / 2
            HL_mod[r0 : r0 + bsz, c0 : c0 + bsz] = _idct2d(dct_block)
        Y_new = _haar_idwt2d(LL, LH, HL_mod, HH)
        diff = Y_new - Y
        for ch in range(3):
            w = [0.299, 0.587, 0.114][ch]
            arr[:, :, ch] = np.clip(arr[:, :, ch] + diff * (w / (w + 1e-8)), 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    def extract(self, img: Image.Image) -> np.ndarray:
        arr = np.array(img).astype(np.float64)
        Y = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        _, _, HL, _ = _haar_dwt2d(Y)
        bsz = 16
        bits = np.zeros(self.payload_bits, dtype=np.int32)
        for k in range(self.payload_bits):
            bi, bj, fi, fj = self.positions[k]
            r0, c0 = bi * bsz, bj * bsz
            block = HL[r0 : r0 + bsz, c0 : c0 + bsz]
            dct_block = _dct2d(block)
            c_val = dct_block[fi, fj]
            q0 = np.round(c_val / self.delta) * self.delta
            q1 = q0 + self.delta / 2
            bits[k] = 0 if abs(c_val - q0) < abs(c_val - q1) else 1
        return bits

    def eval_accuracy(self, img: Image.Image) -> float:
        extracted = self.extract(img)
        return float(np.mean(extracted == self.wm_bits))
