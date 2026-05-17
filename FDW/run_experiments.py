"""
FDW Complete Experiment Suite
按 docs/实验设计文档 产出到全部实验表格和图。

Usage:
    python run_experiments.py baseline         --N 1000
    python run_experiments.py ablation         --N 1000
    python run_experiments.py sweep            --N 1000
    python run_experiments.py geometric        --N 1000
    python run_experiments.py quality          --N 1000
    python run_experiments.py lambda_sweep     --N 1000
    python run_experiments.py gamma_sweep      --N 1000
    python run_experiments.py tstar_sweep      --N 1000
    python run_experiments.py capacity_sweep   --N 1000
    python run_experiments.py fig1
    python run_experiments.py fig5
    python run_experiments.py fig6
    python run_experiments.py all              --N 1000

Output directory: experiment_output/
"""

import argparse
import os
import sys
import json
import time
from statistics import mean, stdev
from itertools import product

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
GS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Gaussian-Shading')
sys.path.insert(0, GS_DIR)

from diffusers import DDIMScheduler
from image_utils import set_random_seed, transform_img, measure_similarity
from watermark import Gaussian_Shading, Gaussian_Shading_chacha
from fdw_pipeline import FDWStableDiffusionPipeline, InversableStableDiffusionPipeline
from watermark_fdw import FDW_Watermark
from attacks import get_attack, ATTACK_REGISTRY, ATTACK_GROUPS
from baseline_dwdct import DwtDctWatermark

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'experiment_output')
_LOG_FILES = []


def _log(text='', end='\n'):
    print(text, end=end)
    for f in _LOG_FILES:
        f.write(text + end)


def _open_log(out_dir, name='result'):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f'{name}.txt')
    f = open(p, 'a', encoding='utf-8')
    _LOG_FILES.append(f)
    return f


def _close_log(f):
    if f in _LOG_FILES:
        _LOG_FILES.remove(f)
    f.close()


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

_HIGH_QUALITY_PROMPTS = [
    "a serene mountain lake at sunset, with golden light reflecting on calm water, snow-capped peaks in the background, photorealistic",
    "a cozy japanese ramen shop at night, warm lantern light, steam rising from bowls, detailed anime style",
    "a vast lavender field in Provence under a clear blue sky, rows of purple flowers stretching to the horizon",
    "a young woman with long black hair standing under cherry blossom trees, petals falling, soft lighting, anime illustration",
    "a lighthouse on a rocky cliff during a dramatic storm, crashing waves, dark clouds, cinematic lighting",
    "a quiet narrow alley in an old European town, cobblestones, flower boxes on windows, morning sunlight",
    "a futuristic city skyline at dusk, neon signs reflecting in wet streets, cyberpunk atmosphere",
    "a cute cat sleeping on a sunlit windowsill, soft bokeh background, warm afternoon light",
    "a misty bamboo forest path, soft diffused light filtering through tall bamboo, peaceful atmosphere",
    "an astronaut floating above Earth, stars visible in the background, photorealistic, detailed spacesuit",
    "a vintage coffee shop interior, exposed brick walls, warm lighting, latte art on a wooden table",
    "a girl riding a bicycle along a seaside road, ocean breeze, bright summer day, studio ghibli style",
    "a frozen waterfall in winter, icicles glistening, snow-covered rocks, soft blue tones",
    "a colorful hot air balloon festival at dawn, dozens of balloons over a green valley",
    "a samurai standing in a field of tall grass, wind blowing, dramatic sunset, digital painting",
    "a small sailboat on a turquoise tropical sea, white sand beach in the distance, aerial view",
    "a wolf howling on a snow-covered ridge under the northern lights, starry sky",
    "a bustling Tokyo street at night, rain-slicked pavement reflecting neon signs, photorealistic",
    "a child reaching up to catch a butterfly in a sunlit meadow, soft pastel colors, illustration",
    "a steaming cup of tea on a wooden tray beside a window overlooking a rainy garden, peaceful mood",
    "a grand cathedral interior with stained glass windows casting colorful light, dramatic perspective",
    "a dragon perched on a castle tower at sunset, fantasy illustration, detailed scales and wings",
    "a quiet library with floor-to-ceiling bookshelves, warm lamp light, cozy reading nook",
    "a speedpainting of a red sports car on a winding mountain road, motion blur, golden hour",
    "a young mage casting a glowing spell in a dark forest, magical particles, fantasy art",
    "a tranquil koi pond in a Japanese garden, lily pads, arched wooden bridge, autumn leaves",
    "a rustic cabin in the woods during autumn, golden and red leaves, smoke from chimney",
    "a girl with silver hair sitting on a rooftop looking at the city lights, anime style, melancholic mood",
    "a herd of wild horses galloping across a dusty plain, dramatic backlighting, cinematic",
    "a coral reef seen through crystal clear water, tropical fish, sunlight beams, vibrant colors",
    "a medieval blacksmith forge, glowing molten metal, sparks flying, dramatic lighting",
    "a vintage train winding through snow-covered mountains, steam rising, panoramic view",
    "a bowl of fresh ramen with chashu and a soft-boiled egg, steam rising, food photography",
    "a magical forest with bioluminescent mushrooms and floating fireflies, fantasy atmosphere",
    "a woman in a red dress walking through an art gallery, elegant, soft museum lighting",
    "a snowy village at twilight, warm windows glowing, Christmas lights, peaceful winter scene",
    "a phoenix rising from flames against a dark sky, vibrant orange and gold feathers, epic",
    "a peaceful rice terrace at sunrise, layers of green fields reflecting morning light, Bali",
    "a boy and his dog watching the sunset from a hilltop, silhouette, warm colors, nostalgic",
    "a deep space nebula with vibrant purple and blue gas clouds, stars, photorealistic",
    "a street musician playing saxophone under a streetlamp, rain, film noir atmosphere",
    "a majestic white tiger resting in dense jungle foliage, dappled sunlight, wildlife photography",
    "a row of colorful houses along a canal in Amsterdam, bicycles parked outside, spring flowers",
    "a young girl holding a glowing lantern in a dark forest path, magical realism, detailed",
    "a vast desert landscape with towering sand dunes, dramatic shadows, golden hour photography",
    "an old bookshop with towering shelves and a rolling ladder, warm afternoon light streaming in",
    "a pirate ship sailing through a stormy sea, lightning illuminating dark waves, dramatic",
    "a rooftop garden in a modern city, green plants contrasting with glass buildings, urban oasis",
    "a baby elephant playing in a river, splashing water, joyful, wildlife photography, golden light",
]


class _PromptDataset:
    def __init__(self, prompts):
        self.prompts = prompts

    def __getitem__(self, idx):
        return {'prompt': self.prompts[idx % len(self.prompts)]}

    def __len__(self):
        return len(self.prompts)


def get_dataset():
    return _PromptDataset(_HIGH_QUALITY_PROMPTS), 'prompt'


def load_pipe(model_path='Manojb/stable-diffusion-2-1-base', method='fdw', device='cuda'):
    sched = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
    cls = FDWStableDiffusionPipeline if method == 'fdw' else InversableStableDiffusionPipeline
    pipe = cls.from_pretrained(model_path, scheduler=sched, torch_dtype=torch.float16)
    pipe.safety_checker = None
    return pipe.to(device)


def make_fdw_wm():
    return FDW_Watermark(ch_factor=1, hw_factor=4, fpr=1e-6, user_number=1_000_000,
                         payload_bits=512, use_ecc=True, lambda_freq=0.02, alpha_max=0.0,
                         t_start=0.2, t_end=0.6, use_fd_detect=False, use_spatial_perm=False)


def make_gs_wm():
    return Gaussian_Shading_chacha(1, 8, 1e-6, 1_000_000)


DEFAULT_ATTACKS = [
    'clean', 'jpeg_75', 'jpeg_50', 'jpeg_25', 'gauss_blur_4', 'gauss_noise_005',
    'crop_080', 'crop_060', 'rotate_15', 'rotate_45', 'scale_075', 'resize_025',
    'brightness_2', 'color_jitter', 'adversarial_8', 'stirmark_rst', 'stirmark_all',
]


# ── Image generation ──────────────────────────────────────────────────────────

def generate_fdw_images(pipe, dataset, prompt_key, N, device):
    text_emb = pipe.get_text_embedding('')
    images, watermarks = [], []
    for i in range(N):
        set_random_seed(i)
        wm = make_fdw_wm()
        init_w = wm.create_watermark_and_return_w()
        fdsc_fft = wm.get_fdsc_template()
        tmask, tpat = wm.get_x_template()
        out = pipe(
            dataset[i][prompt_key], num_images_per_prompt=1, guidance_scale=7.5,
            num_inference_steps=50, height=512, width=512, latents=init_w,
            fdsc_template_fft=fdsc_fft, fdsc_t_start=0.2, fdsc_t_end=0.6, fdsc_alpha_max=0.0,
            template_mask=tmask, template_pattern=tpat, template_t_start=0.3,
            template_t_end=1.0, template_gamma=4.0, template_mode='shallow',
        )
        images.append(out.images[0])
        watermarks.append(wm)
    return images, watermarks, text_emb


def generate_gs_images(pipe, dataset, prompt_key, N, device):
    text_emb = pipe.get_text_embedding('')
    images, watermarks = [], []
    for i in range(N):
        set_random_seed(i)
        wm = make_gs_wm()
        init_w = wm.create_watermark_and_return_w()
        out = pipe(
            dataset[i][prompt_key], num_images_per_prompt=1, guidance_scale=7.5,
            num_inference_steps=50, height=512, width=512, latents=init_w,
        )
        images.append(out.images[0])
        watermarks.append(wm)
    return images, watermarks, text_emb


def generate_clean_images(pipe, dataset, prompt_key, N, device):
    from tqdm import tqdm
    text_emb = pipe.get_text_embedding('')
    images = []
    for i in tqdm(range(N), desc='[Clean] Generating'):
        set_random_seed(i)
        z = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
        out = pipe(
            dataset[i % len(dataset)][prompt_key], num_images_per_prompt=1,
            guidance_scale=7.5, num_inference_steps=50, height=512, width=512, latents=z,
        )
        images.append(out.images[0])
    return images, text_emb


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_fdw_gs(attack_name, images, watermarks, pipe, text_emb, N, device,
                geo_correct=True):
    attack_fn = get_attack(attack_name)
    acc_list = []
    for img, wm in zip(images, watermarks):
        attacked = attack_fn(img)
        if geo_correct and isinstance(wm, FDW_Watermark):
            from attacks import detect_and_correct_geom
            img_t_tensor = transform_img(attacked).unsqueeze(0).to(text_emb.dtype).to(device)
            latents_raw = pipe.get_image_latents(img_t_tensor, sample=False)
            zT_raw = pipe.forward_diffusion(
                latents=latents_raw, text_embeddings=text_emb,
                guidance_scale=1, num_inference_steps=50,
            )

            def _score_fn(candidate_img, _wm=wm, _pipe=pipe, _te=text_emb, _dev=device):
                _it = transform_img(candidate_img).unsqueeze(0).to(text_emb.dtype).to(_dev)
                _lat = _pipe.get_image_latents(_it, sample=False)
                _rev = _pipe.forward_diffusion(
                    latents=_lat, text_embeddings=_te,
                    guidance_scale=1, num_inference_steps=50,
                )
                return _wm.score_watermark(_rev)

            corrected, _, _, _ = detect_and_correct_geom(
                zT_raw, attacked, 512, score_fn=_score_fn)
            img_t = transform_img(corrected).unsqueeze(0).to(text_emb.dtype).to(device)
        else:
            img_t = transform_img(attacked).unsqueeze(0).to(text_emb.dtype).to(device)

        latents = pipe.get_image_latents(img_t, sample=False)
        rev = pipe.forward_diffusion(
            latents=latents, text_embeddings=text_emb,
            guidance_scale=1, num_inference_steps=50,
        )
        acc = wm.eval_watermark(rev)
        acc_list.append(acc)

    tpr_det = sum(getattr(wm, 'tp_onebit_count', 0) for wm in watermarks)
    tpr_trace = sum(getattr(wm, 'tp_bits_count', 0) for wm in watermarks)
    for wm in watermarks:
        wm.tp_onebit_count = 0
        wm.tp_bits_count = 0

    return {
        'tpr_detection': tpr_det / N,
        'tpr_traceability': tpr_trace / N,
        'mean_acc': mean(acc_list),
        'std_acc': stdev(acc_list) if len(acc_list) > 1 else 0.0,
    }


def eval_dwdct(attack_name, images, dwdct_list, N):
    attack_fn = get_attack(attack_name)
    acc_list = []
    for img, dwdct in zip(images, dwdct_list):
        wm_img = dwdct.embed(img)
        attacked = attack_fn(wm_img)
        acc = dwdct.eval_accuracy(attacked)
        acc_list.append(acc)
    return {
        'tpr_detection': 0.0,
        'tpr_traceability': 0.0,
        'mean_acc': mean(acc_list),
        'std_acc': stdev(acc_list) if len(acc_list) > 1 else 0.0,
    }


# ── Table / Plot helpers ─────────────────────────────────────────────────────

def print_table1(results):
    print('\n' + '=' * 100)
    print('Table 1: Baseline Comparison (TPR Detection / TPR Traceability / Bit Accuracy)')
    print('=' * 100)
    header = f"{'Attack':<20} {'FDW':>25} {'GS':>25} {'DwtDct':>25}"
    print(header)
    print('-' * 100)
    for atk in DEFAULT_ATTACKS:
        row = f"{atk:<20}"
        for method in ['fdw', 'gs', 'dwdct']:
            if method in results and atk in results[method]:
                r = results[method][atk]
                row += f"  {r['tpr_detection']:.2f}/{r['tpr_traceability']:.2f}/{r['mean_acc']:.4f}"
            else:
                row += f"  {'N/A':>25}"
        print(row)
    print('=' * 100)


def save_table1_summary(baseline_results, quality_results, out_dir):
    lines = []
    lines.append('=' * 100)
    lines.append('Table 1 (Summary): Per-Method Aggregated Results')
    lines.append('=' * 100)
    header = f"{'Method':<15} {'TPR(Clean)':>12} {'TPR(Adv)':>12} {'Acc(Clean)':>12} {'Acc(Adv)':>12} {'CLIP-Score':>12}"
    lines.append(header)
    lines.append('-' * len(header))
    adv_attacks = [a for a in DEFAULT_ATTACKS if a != 'clean']
    for method in ['fdw', 'gs', 'dwdct']:
        if method not in baseline_results:
            continue
        clean_r = baseline_results[method].get('clean', {})
        adv_accs = [baseline_results[method][a]['mean_acc'] for a in adv_attacks
                    if a in baseline_results[method]]
        adv_tprs = [baseline_results[method][a]['tpr_detection'] for a in adv_attacks
                    if a in baseline_results[method]]
        clip_s = quality_results.get(method, {}).get('mean', None)
        clip_str = f"{clip_s:.4f}" if clip_s is not None else "N/A"
        row = (f"{method:<15} "
               f"{clean_r.get('tpr_detection', 0):>12.4f} "
               f"{mean(adv_tprs):>12.4f} "
               f"{clean_r.get('mean_acc', 0):>12.4f} "
               f"{mean(adv_accs):>12.4f} "
               f"{clip_str:>12}")
        lines.append(row)
    lines.append('=' * 100)
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(out_dir, 'table1_summary.txt'), 'w') as f:
        f.write(text)


def save_table2(results, out_dir):
    lines = []
    lines.append('=' * 100)
    lines.append('Table 2: Per-Attack Detailed Comparison')
    lines.append('Format: TPR_Det / TPR_Trace / Bit_Acc')
    lines.append('=' * 100)
    header = f"{'Attack':<20} {'FDW':>25} {'GS':>25} {'DwtDct':>25}"
    lines.append(header)
    lines.append('-' * 100)
    for atk in DEFAULT_ATTACKS:
        row = f"{atk:<20}"
        for method in ['fdw', 'gs', 'dwdct']:
            if method in results and atk in results[method]:
                r = results[method][atk]
                row += f"  {r['tpr_detection']:.2f}/{r['tpr_traceability']:.2f}/{r['mean_acc']:.4f}"
            else:
                row += f"  {'N/A':>25}"
        lines.append(row)
    lines.append('-' * 100)
    avg_row = f"{'Average':<20}"
    for method in ['fdw', 'gs', 'dwdct']:
        if method in results:
            accs = [results[method][a]['mean_acc'] for a in DEFAULT_ATTACKS if a in results[method]]
            tprs = [results[method][a]['tpr_detection'] for a in DEFAULT_ATTACKS if a in results[method]]
            tprt = [results[method][a]['tpr_traceability'] for a in DEFAULT_ATTACKS if a in results[method]]
            avg_row += f"  {mean(tprs):.2f}/{mean(tprt):.2f}/{mean(accs):.4f}"
        else:
            avg_row += f"  {'N/A':>25}"
    lines.append(avg_row)
    lines.append('=' * 100)
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(out_dir, 'table2_per_attack.txt'), 'w') as f:
        f.write(text)


def print_ablation_table(results, configs, attacks):
    print('\n' + '=' * 100)
    print('Table 4: Component Ablation')
    print('=' * 100)
    adversarial_attacks = [a for a in attacks if a != 'clean']
    header = f"{'Config':<35}" + "".join(f"{atk:>15}" for atk in attacks) + f"  {'Avg(Adv)':>15}"
    print(header)
    print('-' * len(header))
    for cfg_name in configs:
        row = f"{cfg_name:<35}"
        adv_accs = []
        for atk in attacks:
            if cfg_name in results and atk in results[cfg_name]:
                acc = results[cfg_name][atk]['mean_acc']
                row += f"{acc:>15.4f}"
                if atk != 'clean':
                    adv_accs.append(acc)
            else:
                row += f"{'N/A':>15}"
        if adv_accs:
            row += f"  {mean(adv_accs):>15.4f}"
        else:
            row += f"  {'N/A':>15}"
        print(row)
    print('=' * len(header))


def plot_baseline_line(results, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    attacks = DEFAULT_ATTACKS
    methods = ['fdw', 'gs', 'dwdct']
    labels = ['FDW (Ours)', 'Gaussian Shading', 'DwtDct']
    colors = ['#7B2D8B', '#C8A951', '#4A90D9']
    markers = ['o', 's', '^']
    x = np.arange(len(attacks))
    fig, ax = plt.subplots(figsize=(18, 6))
    for m, l, c, mk in zip(methods, labels, colors, markers):
        vals = [results[m][a]['mean_acc'] if m in results and a in results[m] else 0 for a in attacks]
        ax.plot(x, vals, color=c, marker=mk, label=l, linewidth=2, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Bit Accuracy')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_baseline.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_sweep(sweep_results, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    methods = list(sweep_results.keys())
    atk_types = list(sweep_results[methods[0]].keys()) if methods else []
    n = len(atk_types)
    cols = 4
    rows = (n + cols - 1) // cols
    if rows > 1 and n % cols != 0:
        rows += 1
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    colors = {'fdw': '#7B2D8B', 'gs': '#C8A951'}
    for idx, atk_type in enumerate(atk_types):
        ax = axes[idx]
        for method in methods:
            data = sweep_results[method].get(atk_type, {})
            if not data:
                continue
            params = sorted(data.keys(), key=lambda x: float(x))
            display = [_SWEEP_DISPLAY.get(atk_type, {}).get(p, p) for p in params]
            acc_vals = [data[p].get('mean_acc', 0) for p in params]
            ax.plot(range(len(params)), acc_vals,
                    color=colors.get(method, 'b'), marker='o',
                    label=f'{method.upper()} Acc', linewidth=2)
            tpr_vals = [data[p].get('mean_tpr', 0) for p in params]
            if any(v > 0 for v in tpr_vals):
                ax.plot(range(len(params)), tpr_vals,
                        color=colors.get(method, 'b'), marker='o',
                        linestyle='--', alpha=0.6,
                        label=f'{method.upper()} TPR', linewidth=1.5)
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels(display, rotation=45, fontsize=8)
        ax.set_title(atk_type)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Bit Accuracy / TPR')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_sweep.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_figure4_geom(geom_results, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for method, color, marker in [('fdw', '#7B2D8B', 'o'), ('gs', '#C8A951', 's')]:
        if 'rotate' in geom_results:
            angles = sorted(geom_results['rotate'].keys(), key=lambda x: float(x))
            vals = [geom_results['rotate'][a][method]['mean_acc'] for a in angles]
            ax1.plot(angles, vals, color=color, marker=marker, label=method.upper(), linewidth=2)
        if 'scale' in geom_results:
            ratios = sorted(geom_results['scale'].keys(), key=lambda x: float(x))
            vals = [geom_results['scale'][r][method]['mean_acc'] for r in ratios]
            ax2.plot(ratios, vals, color=color, marker=marker, label=method.upper(), linewidth=2)
    ax1.set_xlabel('Rotation Angle (°)')
    ax1.set_ylabel('Bit Accuracy')
    ax1.set_title('Rotation Attack')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Scale Ratio')
    ax2.set_ylabel('Bit Accuracy')
    ax2.set_title('Scale Attack')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_geometric.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Baseline Comparison (Table 1, Table 2, Figure 3)
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline(N):
    from tqdm import tqdm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, prompt_key = get_dataset()
    out_dir = os.path.join(OUTPUT_DIR, 'baseline')
    os.makedirs(out_dir, exist_ok=True)
    lf = _open_log(out_dir, 'baseline')

    all_results = {}
    for method in ['fdw', 'gs', 'dwdct']:
        _log(f"\n{'=' * 60}\n[{method.upper()}] Generating {N} images\n{'=' * 60}")
        if method == 'dwdct':
            pipe = load_pipe(device=device)
            clean_images, _ = generate_clean_images(pipe, dataset, prompt_key, N, device)
            dwdct_list = [DwtDctWatermark() for _ in range(N)]
            method_results = {}
            for atk in tqdm(DEFAULT_ATTACKS, desc=f'[{method.upper()}] Attacks'):
                r = eval_dwdct(atk, clean_images, dwdct_list, N)
                method_results[atk] = r
                _log(f"  {atk}: Acc={r['mean_acc']:.4f}")
            del pipe
            torch.cuda.empty_cache()
        else:
            pipe = load_pipe(device=device, method=method)
            if method == 'fdw':
                imgs, wms, te = generate_fdw_images(pipe, dataset, prompt_key, N, device)
            else:
                imgs, wms, te = generate_gs_images(pipe, dataset, prompt_key, N, device)
            method_results = {}
            for atk in tqdm(DEFAULT_ATTACKS, desc=f'[{method.upper()}] Attacks'):
                r = eval_fdw_gs(atk, imgs, wms, pipe, te, N, device,
                                geo_correct=(method == 'fdw'))
                method_results[atk] = r
                _log(f"  {atk}: TPR={r['tpr_detection']:.4f} Acc={r['mean_acc']:.4f}")
            del pipe
            torch.cuda.empty_cache()

        all_results[method] = method_results

    with open(os.path.join(out_dir, 'baseline_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print_table1(all_results)
    save_table2(all_results, out_dir)
    plot_baseline_line(all_results, out_dir)

    quality_path = os.path.join(OUTPUT_DIR, 'quality', 'quality_results.json')
    if os.path.exists(quality_path):
        with open(quality_path) as f:
            quality_results = json.load(f)
        save_table1_summary(all_results, quality_results, out_dir)
    else:
        _log("\n[INFO] Run 'quality' experiment first to generate Table 1 summary with CLIP scores")

    _close_log(lf)
    _log(f"\n[Saved → {out_dir}/]")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Ablation (Table 3, Table 4)
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    'GS baseline (hw=8, 256b)': dict(hw=8, payload=256, ecc=False, fdinit=False,
                                       template=False, geocorrect=False, spatial_perm=False),
    'Expand (hw=4, 1024b)':      dict(hw=4, payload=1024, ecc=False, fdinit=False,
                                       template=False, geocorrect=False, spatial_perm=False),
    '+ FD-Init':                  dict(hw=4, payload=512, ecc=True, fdinit=True,
                                       template=False, geocorrect=False, spatial_perm=False),
    '+ Repetition Code':          dict(hw=4, payload=512, ecc=True, fdinit=True,
                                       template=False, geocorrect=False, spatial_perm=False),
    '+ X-Template (no correct)':  dict(hw=4, payload=512, ecc=True, fdinit=True,
                                       template=True, geocorrect=False, spatial_perm=False),
    'FDW Full':                   dict(hw=4, payload=512, ecc=True, fdinit=True,
                                       template=True, geocorrect=True, spatial_perm=False),
}

ABLATION_ATTACKS = ['clean', 'jpeg_25', 'gauss_blur_4', 'crop_060',
                     'rotate_15', 'rotate_45', 'scale_075']


def run_ablation(N):
    from tqdm import tqdm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, prompt_key = get_dataset()
    out_dir = os.path.join(OUTPUT_DIR, 'ablation')
    os.makedirs(out_dir, exist_ok=True)
    lf = _open_log(out_dir, 'ablation')

    all_results = {}
    for cfg_name, cfg in ABLATION_CONFIGS.items():
        _log(f"\n--- Ablation: {cfg_name} ---")
        pipe = load_pipe(device=device)
        text_emb = pipe.get_text_embedding('')
        wm_cls = (Gaussian_Shading_chacha if cfg['hw'] == 8 else None)

        imgs, wms = [], []
        for i in range(N):
            set_random_seed(i)
            if cfg['hw'] == 8:
                wm = Gaussian_Shading_chacha(1, 8, 1e-6, 1_000_000)
                init_w = wm.create_watermark_and_return_w()
                out = pipe(dataset[i][prompt_key], num_images_per_prompt=1,
                           guidance_scale=7.5, num_inference_steps=50,
                           height=512, width=512, latents=init_w)
            else:
                wm = FDW_Watermark(ch_factor=1, hw_factor=4, fpr=1e-6,
                                   user_number=1_000_000, payload_bits=cfg['payload'],
                                   use_ecc=cfg['ecc'],
                                   lambda_freq=(0.08 if cfg['fdinit'] else 0.0),
                                   alpha_max=0.0, t_start=0.2, t_end=0.6,
                                   use_fd_detect=False,
                                   use_spatial_perm=cfg.get('spatial_perm', False))
                init_w = wm.create_watermark_and_return_w()
                fdsc_fft = wm.get_fdsc_template() if cfg['fdinit'] else None
                tmask, tpat = (wm.get_x_template() if cfg['template'] else (None, None))
                out = pipe(
                    dataset[i][prompt_key], num_images_per_prompt=1,
                    guidance_scale=7.5, num_inference_steps=50,
                    height=512, width=512, latents=init_w,
                    fdsc_template_fft=fdsc_fft, fdsc_t_start=0.2, fdsc_t_end=0.6,
                    fdsc_alpha_max=0.0,
                    template_mask=tmask, template_pattern=tpat,
                    template_t_start=0.3, template_t_end=1.0,
                    template_gamma=4.0, template_mode='shallow',
                )
            imgs.append(out.images[0])
            wms.append(wm)

        cfg_results = {}
        for atk in tqdm(ABLATION_ATTACKS, desc=cfg_name[:30]):
            r = eval_fdw_gs(atk, imgs, wms, pipe, text_emb, N, device,
                            geo_correct=cfg['geocorrect'])
            cfg_results[atk] = r
        all_results[cfg_name] = cfg_results

        with open(os.path.join(out_dir, 'ablation_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        del pipe
        torch.cuda.empty_cache()

    print_ablation_table(all_results, ABLATION_CONFIGS.keys(), ABLATION_ATTACKS)
    _close_log(lf)
    _log(f"\n[Saved → {out_dir}/]")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Attack Intensity Sweep (Figure 2)
# ══════════════════════════════════════════════════════════════════════════════

SWEEP_PARAMS = {
    'jpeg':          {'values': ['90', '75', '50', '35', '25', '15', '10']},
    'gauss_blur':    {'values': ['2', '4', '6', '8', '10']},
    'gauss_noise':   {'values': ['001', '005', '010', '015', '020', '030', '040']},
    'brightness':    {'values': ['2', '4', '6', '8', '10', '12', '14', '16']},
    'rotate':        {'values': ['5', '10', '15', '20', '30', '45', '60', '90']},
    'scale':         {'values': ['090', '080', '075', '060', '050']},
    'crop':          {'values': ['090', '070', '050', '030', '010']},
    'resize':        {'values': ['050', '025', '010']},
}

_SWEEP_DISPLAY = {
    'crop':        {'090': '0.9', '070': '0.7', '050': '0.5', '030': '0.3', '010': '0.1'},
    'gauss_noise': {'001': '0.01', '005': '0.05', '010': '0.10', '015': '0.15',
                    '020': '0.20', '030': '0.30', '040': '0.40'},
    'scale':       {'090': '0.9', '080': '0.8', '075': '0.75', '060': '0.6', '050': '0.5'},
}


def run_sweep(N):
    from tqdm import tqdm
    from PIL import ImageFilter, ImageEnhance
    from attacks import attack_gaussian_noise, attack_scale, attack_crop

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, prompt_key = get_dataset()
    out_dir = os.path.join(OUTPUT_DIR, 'sweep')
    os.makedirs(out_dir, exist_ok=True)
    lf = _open_log(out_dir, 'sweep')

    sweep_results = {}

    for method in ['fdw', 'gs']:
        _log(f"\n{'=' * 60}\n[Sweep] {method.upper()}\n{'=' * 60}")
        pipe = load_pipe(device=device, method=method)
        if method == 'fdw':
            imgs, wms, te = generate_fdw_images(pipe, dataset, prompt_key, N, device)
        else:
            imgs, wms, te = generate_gs_images(pipe, dataset, prompt_key, N, device)

        method_sweep = {}
        for atk_type, spec in SWEEP_PARAMS.items():
            type_results = {}
            for val in tqdm(spec['values'], desc=f'  {atk_type}'):
                atk_name = f'{atk_type}_{val}'
                try:
                    attack_fn = get_attack(atk_name)
                except KeyError:
                    _log(f"\n  [WARN] Unknown attack '{atk_name}' — skipping!")
                    continue
                acc_list = []
                tpr_list = []
                for img, wm in zip(imgs, wms):
                    attacked = attack_fn(img)
                    img_t = transform_img(attacked).unsqueeze(0).to(te.dtype).to(device)
                    if method == 'fdw':
                        from attacks import detect_and_correct_geom
                        raw_lat = pipe.get_image_latents(img_t, sample=False)
                        zT_raw = pipe.forward_diffusion(latents=raw_lat, text_embeddings=te,
                                                        guidance_scale=1, num_inference_steps=50)

                        def _score_fn(candidate_img, _wm=wm, _pipe=pipe, _te=te, _dev=device):
                            _it = transform_img(candidate_img).unsqueeze(0).to(te.dtype).to(_dev)
                            _lat = _pipe.get_image_latents(_it, sample=False)
                            _rev = _pipe.forward_diffusion(
                                latents=_lat, text_embeddings=_te,
                                guidance_scale=1, num_inference_steps=50)
                            return _wm.score_watermark(_rev)

                        corrected, _, _, _ = detect_and_correct_geom(
                            zT_raw, attacked, 512, score_fn=_score_fn)
                        img_t = transform_img(corrected).unsqueeze(0).to(te.dtype).to(device)
                    latents = pipe.get_image_latents(img_t, sample=False)
                    rev = pipe.forward_diffusion(latents=latents, text_embeddings=te,
                                                 guidance_scale=1, num_inference_steps=50)
                    acc_list.append(wm.eval_watermark(rev))
                    tpr_list.append(1.0 if getattr(wm, 'tp_onebit_count', 0) > 0 else 0.0)
                    if hasattr(wm, 'tp_onebit_count'):
                        wm.tp_onebit_count = 0
                    if hasattr(wm, 'tp_bits_count'):
                        wm.tp_bits_count = 0
                type_results[val] = {'mean_acc': mean(acc_list), 'mean_tpr': mean(tpr_list)}
            method_sweep[atk_type] = type_results

        sweep_results[method] = method_sweep
        del pipe
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, 'sweep_results.json'), 'w') as f:
        json.dump(sweep_results, f, indent=2)
    plot_sweep(sweep_results, out_dir)
    _close_log(lf)
    _log(f"\n[Saved → {out_dir}/]")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 4: Geometric Attack Detail (Figure 4)
# ══════════════════════════════════════════════════════════════════════════════

def run_geometric(N):
    from tqdm import tqdm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, prompt_key = get_dataset()
    out_dir = os.path.join(OUTPUT_DIR, 'geometric')
    os.makedirs(out_dir, exist_ok=True)
    lf = _open_log(out_dir, 'geometric')

    rotate_angles = ['5', '10', '15', '20', '30', '45', '60', '90']
    scale_ratios = ['0.95', '0.90', '0.85', '0.80', '0.75', '0.70', '0.65',
                    '0.60', '0.55', '0.50']

    geom_results = {'rotate': {}, 'scale': {}}

    for method in ['fdw', 'gs']:
        _log(f"\n{'=' * 60}\n[Geometric] {method.upper()}\n{'=' * 60}")
        pipe = load_pipe(device=device, method=method)
        if method == 'fdw':
            imgs, wms, te = generate_fdw_images(pipe, dataset, prompt_key, N, device)
        else:
            imgs, wms, te = generate_gs_images(pipe, dataset, prompt_key, N, device)

        for angle in tqdm(rotate_angles, desc='Rotation'):
            atk_name = f'rotate_{angle}'
            try:
                get_attack(atk_name)
            except KeyError:
                continue
            r = eval_fdw_gs(atk_name, imgs, wms, pipe, te, N, device,
                            geo_correct=(method == 'fdw'))
            if angle not in geom_results['rotate']:
                geom_results['rotate'][angle] = {}
            geom_results['rotate'][angle][method] = r

        for ratio in tqdm(scale_ratios, desc='Scale'):
            atk_name = f'scale_{int(float(ratio) * 100):03d}'
            try:
                attack_fn = get_attack(atk_name)
            except KeyError:
                continue
            r = eval_fdw_gs(atk_name, imgs, wms, pipe, te, N, device,
                            geo_correct=(method == 'fdw'))
            if ratio not in geom_results['scale']:
                geom_results['scale'][ratio] = {}
            geom_results['scale'][ratio][method] = r

        del pipe
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, 'geometric_results.json'), 'w') as f:
        json.dump(geom_results, f, indent=2)
    plot_figure4_geom(geom_results, out_dir)
    _close_log(lf)
    _log(f"\n[Saved → {out_dir}/]")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 5: FD-Init Lambda Sweep (§7.2)
# ══════════════════════════════════════════════════════════════════════════════

LAMBDA_VALUES = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20]
LAMBDA_ATTACKS = ['clean', 'jpeg_25', 'gauss_blur_4', 'gauss_noise_005',
                  'crop_060', 'rotate_15', 'rotate_45', 'scale_075']


def run_lambda_sweep(N):
    from tqdm import tqdm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, prompt_key = get_dataset()
    out_dir = os.path.join(OUTPUT_DIR, 'lambda_sweep')
    os.makedirs(out_dir, exist_ok=True)
    lf = _open_log(out_dir, 'lambda_sweep')

    results = {}
    for lam in LAMBDA_VALUES:
        tag = f'lambda={lam:.2f}'
        _log(f"\n{'=' * 60}\n[λ Sweep] {tag}\n{'=' * 60}")
        pipe = load_pipe(device=device)
        text_emb = pipe.get_text_embedding('')
        imgs, wms = [], []
        for i in range(N):
            set_random_seed(i)
            wm = FDW_Watermark(ch_factor=1, hw_factor=4, fpr=1e-6, user_number=1_000_000,
                               payload_bits=512, use_ecc=True, lambda_freq=lam, alpha_max=0.0,
                               t_start=0.2, t_end=0.6, use_fd_detect=False, use_spatial_perm=False)
            init_w = wm.create_watermark_and_return_w()
            fdsc_fft = wm.get_fdsc_template()
            tmask, tpat = wm.get_x_template()
            out = pipe(
                dataset[i][prompt_key], num_images_per_prompt=1, guidance_scale=7.5,
                num_inference_steps=50, height=512, width=512, latents=init_w,
                fdsc_template_fft=fdsc_fft, fdsc_t_start=0.2, fdsc_t_end=0.6,
                fdsc_alpha_max=0.0, template_mask=tmask, template_pattern=tpat,
                template_t_start=0.3, template_t_end=1.0, template_gamma=4.0,
                template_mode='shallow',
            )
            imgs.append(out.images[0])
            wms.append(wm)

        lam_results = {}
        for atk in tqdm(LAMBDA_ATTACKS, desc=tag):
            r = eval_fdw_gs(atk, imgs, wms, pipe, text_emb, N, device, geo_correct=True)
            lam_results[atk] = r
            _log(f"  {atk}: Acc={r['mean_acc']:.4f}")
        results[tag] = lam_results
        del pipe
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, 'lambda_sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    _log('\n' + '=' * 60)
    _log('Table: FD-Init λ Sweep (Bit Accuracy)')
    _log('=' * 60)
    header = f"{'λ':>6s} | " + " | ".join(f"{a:>14s}" for a in LAMBDA_ATTACKS)
    _log(header)
    _log("-" * len(header))
    for tag in results:
        row = f"{tag:>6s} | "
        for atk in LAMBDA_ATTACKS:
            row += f"{results[tag][atk]['mean_acc']:14.4f} | "
        _log(row)

    _plot_param_sweep(results, 'λ', LAMBDA_ATTACKS,
                      os.path.join(out_dir, 'fig_lambda_sweep.png'))
    _close_log(lf)
    _log(f"\n[Saved → {out_dir}/]")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 6: Template Gamma Sweep (§7.3)
# ══════════════════════════════════════════════════════════════════════════════

GAMMA_VALUES = [0, 2, 4, 8, 12, 16]
GAMMA_ATTACKS = ['clean', 'rotate_15', 'rotate_45', 'scale_075']


def run_gamma_sweep(N):
    from tqdm import tqdm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, prompt_key = get_dataset()
    out_dir = os.path.join(OUTPUT_DIR, 'gamma_sweep')
    os.makedirs(out_dir, exist_ok=True)
    lf = _open_log(out_dir, 'gamma_sweep')

    results = {}
    for gamma in GAMMA_VALUES:
        tag = f'gamma={gamma}'
        _log(f"\n{'=' * 60}\n[γ Sweep] {tag}\n{'=' * 60}")
        pipe = load_pipe(device=device)
        text_emb = pipe.get_text_embedding('')
        imgs, wms = [], []
        for i in range(N):
            set_random_seed(i)
            wm = FDW_Watermark(ch_factor=1, hw_factor=4, fpr=1e-6, user_number=1_000_000,
                               payload_bits=512, use_ecc=True, lambda_freq=0.02, alpha_max=0.0,
                               t_start=0.2, t_end=0.6, use_fd_detect=False, use_spatial_perm=False)
            init_w = wm.create_watermark_and_return_w()
            fdsc_fft = wm.get_fdsc_template()
            if gamma > 0:
                tmask, tpat = wm.get_x_template()
            else:
                tmask, tpat = None, None
            out = pipe(
                dataset[i][prompt_key], num_images_per_prompt=1, guidance_scale=7.5,
                num_inference_steps=50, height=512, width=512, latents=init_w,
                fdsc_template_fft=fdsc_fft, fdsc_t_start=0.2, fdsc_t_end=0.6,
                fdsc_alpha_max=0.0, template_mask=tmask, template_pattern=tpat,
                template_t_start=0.3, template_t_end=1.0, template_gamma=float(gamma),
                template_mode='shallow',
            )
            imgs.append(out.images[0])
            wms.append(wm)

        gam_results = {}
        for atk in tqdm(GAMMA_ATTACKS, desc=tag):
            r = eval_fdw_gs(atk, imgs, wms, pipe, text_emb, N, device,
                            geo_correct=(gamma > 0))
            gam_results[atk] = r
            _log(f"  {atk}: Acc={r['mean_acc']:.4f}")
        prompts = [dataset[i % len(dataset)][prompt_key] for i in range(N)]
        clip_score = _clip_for_sweep(imgs, prompts)
        if clip_score is not None:
            gam_results['_clip_score'] = clip_score
            _log(f"  CLIP-Score: {clip_score:.4f}")
        results[tag] = gam_results
        del pipe
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, 'gamma_sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    _log('\n' + '=' * 60)
    _log('Table: Template γ Sweep (Bit Accuracy)')
    _log('=' * 60)
    header = f"{'γ':>8s} | " + " | ".join(f"{a:>14s}" for a in GAMMA_ATTACKS) + " | " + f"{'CLIP-Score':>12s}"
    _log(header)
    _log("-" * len(header))
    for tag in results:
        row = f"{tag:>8s} | "
        for atk in GAMMA_ATTACKS:
            row += f"{results[tag][atk]['mean_acc']:14.4f} | "
        clip = results[tag].get('_clip_score', None)
        row += f"  {clip:>12.4f} | " if clip is not None else f"  {'N/A':>12s} | "
        _log(row)

    _plot_param_sweep(results, 'γ', GAMMA_ATTACKS,
                      os.path.join(out_dir, 'fig_gamma_sweep.png'))
    _close_log(lf)
    _log(f"\n[Saved → {out_dir}/]")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 7: Template Injection Timestep Sweep (§7.4)
# ══════════════════════════════════════════════════════════════════════════════

TSTAR_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
TSTAR_ATTACKS = ['clean', 'rotate_15', 'rotate_45', 'scale_075']


def run_tstar_sweep(N):
    from tqdm import tqdm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, prompt_key = get_dataset()
    out_dir = os.path.join(OUTPUT_DIR, 'tstar_sweep')
    os.makedirs(out_dir, exist_ok=True)
    lf = _open_log(out_dir, 'tstar_sweep')

    results = {}
    for t_star in TSTAR_VALUES:
        tag = f't*={t_star:.1f}'
        _log(f"\n{'=' * 60}\n[t* Sweep] {tag}\n{'=' * 60}")
        pipe = load_pipe(device=device)
        text_emb = pipe.get_text_embedding('')
        imgs, wms = [], []
        for i in range(N):
            set_random_seed(i)
            wm = FDW_Watermark(ch_factor=1, hw_factor=4, fpr=1e-6, user_number=1_000_000,
                               payload_bits=512, use_ecc=True, lambda_freq=0.02, alpha_max=0.0,
                               t_start=0.2, t_end=0.6, use_fd_detect=False, use_spatial_perm=False)
            init_w = wm.create_watermark_and_return_w()
            fdsc_fft = wm.get_fdsc_template()
            tmask, tpat = wm.get_x_template()
            out = pipe(
                dataset[i][prompt_key], num_images_per_prompt=1, guidance_scale=7.5,
                num_inference_steps=50, height=512, width=512, latents=init_w,
                fdsc_template_fft=fdsc_fft, fdsc_t_start=0.2, fdsc_t_end=0.6,
                fdsc_alpha_max=0.0, template_mask=tmask, template_pattern=tpat,
                template_t_start=t_star, template_t_end=1.0, template_gamma=4.0,
                template_mode='shallow',
            )
            imgs.append(out.images[0])
            wms.append(wm)

        ts_results = {}
        for atk in tqdm(TSTAR_ATTACKS, desc=tag):
            r = eval_fdw_gs(atk, imgs, wms, pipe, text_emb, N, device, geo_correct=True)
            ts_results[atk] = r
            _log(f"  {atk}: Acc={r['mean_acc']:.4f}")
        prompts = [dataset[i % len(dataset)][prompt_key] for i in range(N)]
        clip_score = _clip_for_sweep(imgs, prompts)
        if clip_score is not None:
            ts_results['_clip_score'] = clip_score
            _log(f"  CLIP-Score: {clip_score:.4f}")
        results[tag] = ts_results
        del pipe
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, 'tstar_sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    _log('\n' + '=' * 60)
    _log('Table: Template t* Sweep (Bit Accuracy)')
    _log('=' * 60)
    header = f"{'t*':>6s} | " + " | ".join(f"{a:>14s}" for a in TSTAR_ATTACKS) + " | " + f"{'CLIP-Score':>12s}"
    _log(header)
    _log("-" * len(header))
    for tag in results:
        row = f"{tag:>6s} | "
        for atk in TSTAR_ATTACKS:
            row += f"{results[tag][atk]['mean_acc']:14.4f} | "
        clip = results[tag].get('_clip_score', None)
        row += f"  {clip:>12.4f} | " if clip is not None else f"  {'N/A':>12s} | "
        _log(row)

    _plot_param_sweep(results, 't*', TSTAR_ATTACKS,
                      os.path.join(out_dir, 'fig_tstar_sweep.png'))
    _close_log(lf)
    _log(f"\n[Saved → {out_dir}/]")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 8: Capacity-Redundancy Tradeoff (Table 3, §7.1)
# ══════════════════════════════════════════════════════════════════════════════

CAPACITY_CONFIGS = {
    'hw8_256b':               {'hw': 8, 'payload': 256,  'ecc': False, 'fdinit': False, 'use_gs': True},
    'hw4_1024b':              {'hw': 4, 'payload': 1024, 'ecc': False, 'fdinit': False, 'use_gs': False},
    'hw4_512b_repeat':        {'hw': 4, 'payload': 512,  'ecc': True,  'fdinit': False, 'use_gs': False},
    'hw4_512b_repeat_fdinit': {'hw': 4, 'payload': 512,  'ecc': True,  'fdinit': True,  'use_gs': False},
    'hw2_4096b':              {'hw': 2, 'payload': 4096, 'ecc': False, 'fdinit': False, 'use_gs': False},
}

CAPACITY_ATTACKS = ['clean', 'jpeg_25', 'gauss_blur_4', 'crop_060',
                     'rotate_15', 'rotate_45']


def run_capacity_sweep(N):
    from tqdm import tqdm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, prompt_key = get_dataset()
    out_dir = os.path.join(OUTPUT_DIR, 'capacity_sweep')
    os.makedirs(out_dir, exist_ok=True)
    lf = _open_log(out_dir, 'capacity_sweep')

    all_results = {}
    for cfg_name, cfg in CAPACITY_CONFIGS.items():
        _log(f"\n--- Capacity: {cfg_name} ---")
        if cfg['use_gs']:
            pipe = load_pipe(device=device, method='gs')
            imgs, wms, te = generate_gs_images(pipe, dataset, prompt_key, N, device)
        else:
            pipe = load_pipe(device=device, method='fdw')
            text_emb = pipe.get_text_embedding('')
            imgs, wms = [], []
            for i in range(N):
                set_random_seed(i)
                wm = FDW_Watermark(
                    ch_factor=1, hw_factor=cfg['hw'], fpr=1e-6,
                    user_number=1_000_000, payload_bits=cfg['payload'],
                    use_ecc=cfg['ecc'],
                    lambda_freq=(0.08 if cfg['fdinit'] else 0.0),
                    alpha_max=0.0, t_start=0.2, t_end=0.6,
                    use_fd_detect=False, use_spatial_perm=False)
                init_w = wm.create_watermark_and_return_w()
                fdsc_fft = wm.get_fdsc_template() if cfg['fdinit'] else None
                out = pipe(
                    dataset[i][prompt_key], num_images_per_prompt=1,
                    guidance_scale=7.5, num_inference_steps=50,
                    height=512, width=512, latents=init_w,
                    fdsc_template_fft=fdsc_fft, fdsc_t_start=0.2, fdsc_t_end=0.6,
                    fdsc_alpha_max=0.0,
                    template_mask=None, template_pattern=None,
                    template_t_start=0.3, template_t_end=1.0,
                    template_gamma=4.0, template_mode='shallow',
                )
                imgs.append(out.images[0])
                wms.append(wm)
            te = text_emb

        cfg_results = {}
        for atk in tqdm(CAPACITY_ATTACKS, desc=cfg_name):
            r = eval_fdw_gs(atk, imgs, wms, pipe, te, N, device, geo_correct=False)
            cfg_results[atk] = r
            _log(f"  {atk}: Acc={r['mean_acc']:.4f}")
        all_results[cfg_name] = cfg_results

        with open(os.path.join(out_dir, 'capacity_sweep_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        del pipe
        torch.cuda.empty_cache()

    _log('\n' + '=' * 80)
    _log('Table 3: Capacity-Redundancy Tradeoff (Bit Accuracy)')
    _log('=' * 80)
    header = f"{'Attack':<20}" + "".join(f"  {n:>22}" for n in CAPACITY_CONFIGS)
    _log(header)
    _log('-' * len(header))
    for atk in CAPACITY_ATTACKS:
        row = f"{atk:<20}"
        for cn in CAPACITY_CONFIGS:
            if atk in all_results.get(cn, {}):
                row += f"  {all_results[cn][atk]['mean_acc']:>22.4f}"
            else:
                row += f"  {'N/A':>22}"
        _log(row)
    avg_row = f"{'Average':<20}"
    for cn in CAPACITY_CONFIGS:
        accs = [all_results[cn][a]['mean_acc'] for a in CAPACITY_ATTACKS
                if a in all_results.get(cn, {})]
        avg_row += f"  {mean(accs):>22.4f}" if accs else f"  {'N/A':>22}"
    _log(avg_row)
    _log('=' * 80)

    _close_log(lf)
    _log(f"\n[Saved → {out_dir}/]")


def _measure_clip_batch(pipe, prompts, images, ref_model, ref_preprocess, ref_tokenizer, device):
    from image_utils import measure_similarity
    scores = []
    for prompt, img in zip(prompts, images):
        sim = measure_similarity([img], prompt, ref_model, ref_preprocess, ref_tokenizer, device)
        scores.append(sim[0].item())
    return mean(scores)


def _clip_for_sweep(images, prompts, N_sample=10):
    try:
        import open_clip
    except ImportError:
        return None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ref_model, _, ref_preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', device=device)
    ref_tokenizer = open_clip.get_tokenizer('ViT-L-14')
    sample_imgs = images[:N_sample]
    sample_prompts = prompts[:N_sample]
    score = _measure_clip_batch(None, sample_prompts, sample_imgs,
                                ref_model, ref_preprocess, ref_tokenizer, device)
    del ref_model
    torch.cuda.empty_cache()
    return score


# ══════════════════════════════════════════════════════════════════════════════
# Sweep Plot Helpers (for lambda / gamma / tstar)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_param_sweep(results, param_name, attacks, output_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tags = sorted(results.keys())
    x_vals = [float(t.split('=')[1]) for t in tags]
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v', 'x', '*', 'P']
    for idx, atk in enumerate(attacks):
        vals = [results[t][atk]['mean_acc'] for t in tags]
        ax.plot(x_vals, vals, marker=markers[idx % len(markers)],
                label=atk, linewidth=2, markersize=6)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Bit Accuracy')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure Generation Experiments
# ══════════════════════════════════════════════════════════════════════════════

FIG1_ATTACKS = ['clean', 'jpeg_25', 'crop_060', 'gauss_blur_4', 'gauss_noise_005',
                'rotate_15', 'scale_075', 'brightness_2', 'stirmark_rst', 'adversarial_8']


def run_fig1_attack_examples(N=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, prompt_key = get_dataset()
    out_dir = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    pipe = load_pipe(device=device, method='fdw')
    set_random_seed(42)
    prompt = dataset[0][prompt_key]
    wm = make_fdw_wm()
    init_w = wm.create_watermark_and_return_w()
    fdsc_fft = wm.get_fdsc_template()
    tmask, tpat = wm.get_x_template()
    out = pipe(prompt, num_images_per_prompt=1, guidance_scale=7.5,
               num_inference_steps=50, height=512, width=512, latents=init_w,
               fdsc_template_fft=fdsc_fft, fdsc_t_start=0.2, fdsc_t_end=0.6,
               fdsc_alpha_max=0.0, template_mask=tmask, template_pattern=tpat,
               template_t_start=0.3, template_t_end=1.0,
               template_gamma=4.0, template_mode='shallow')
    orig = out.images[0]
    del pipe
    torch.cuda.empty_cache()

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping figure generation")
        return

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    for idx, atk_name in enumerate(FIG1_ATTACKS):
        ax = axes[idx // 5, idx % 5]
        attack_fn = get_attack(atk_name)
        attacked = attack_fn(orig)
        ax.imshow(attacked)
        ax.set_title(atk_name, fontsize=10)
        ax.axis('off')
    plt.suptitle('Figure 1: Attack Examples on Watermarked Image', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig1_attack_examples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved → {out_dir}/fig1_attack_examples.png]")


def run_fig5_visual_quality(N=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, prompt_key = get_dataset()
    out_dir = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    prompts = [dataset[i][prompt_key] for i in range(N)]

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available")
        return

    methods = ['clean', 'fdw', 'gs', 'dwdct']
    method_labels = ['Original', 'FDW (Ours)', 'Gaussian Shading', 'DwtDct']

    all_images = {m: [] for m in methods}
    for col, prompt in enumerate(prompts):
        for method in methods:
            if method == 'dwdct':
                pipe_tmp = load_pipe(device=device, method='fdw')
                set_random_seed(col)
                z = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
                out = pipe_tmp(prompt, num_images_per_prompt=1, guidance_scale=7.5,
                               num_inference_steps=50, height=512, width=512, latents=z)
                clean_img = out.images[0]
                dwdct = DwtDctWatermark()
                wm_img = dwdct.embed(clean_img)
                all_images[method].append(wm_img)
                all_images['clean'].append(clean_img)
                del pipe_tmp
            elif method == 'clean':
                pass
            elif method == 'fdw':
                pipe_tmp = load_pipe(device=device, method='fdw')
                set_random_seed(col)
                wm = make_fdw_wm()
                init_w = wm.create_watermark_and_return_w()
                fdsc_fft = wm.get_fdsc_template()
                tmask, tpat = wm.get_x_template()
                out = pipe_tmp(prompt, num_images_per_prompt=1, guidance_scale=7.5,
                               num_inference_steps=50, height=512, width=512, latents=init_w,
                               fdsc_template_fft=fdsc_fft, fdsc_t_start=0.2, fdsc_t_end=0.6,
                               fdsc_alpha_max=0.0, template_mask=tmask, template_pattern=tpat,
                               template_t_start=0.3, template_t_end=1.0,
                               template_gamma=4.0, template_mode='shallow')
                all_images[method].append(out.images[0])
                del pipe_tmp
            elif method == 'gs':
                pipe_tmp = load_pipe(device=device, method='gs')
                set_random_seed(col)
                wm = make_gs_wm()
                init_w = wm.create_watermark_and_return_w()
                out = pipe_tmp(prompt, num_images_per_prompt=1, guidance_scale=7.5,
                               num_inference_steps=50, height=512, width=512, latents=init_w)
                all_images[method].append(out.images[0])
                del pipe_tmp
            torch.cuda.empty_cache()

    n_rows = len(methods) * 2
    fig, axes = plt.subplots(n_rows, N, figsize=(5 * N, 4 * n_rows))
    if N == 1:
        axes = axes.reshape(n_rows, 1)
    for row_idx, method in enumerate(methods):
        for col in range(N):
            img = all_images[method][col]
            ax_img = axes[row_idx * 2, col]
            ax_img.imshow(img)
            if col == 0:
                ax_img.set_ylabel(method_labels[row_idx], fontsize=10)
            ax_img.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            clean_img = all_images['clean'][col]
            clean_arr = np.array(clean_img).astype(np.float32)
            wm_arr = np.array(img).astype(np.float32)
            residual = np.abs(wm_arr - clean_arr)
            residual = (residual * 10).clip(0, 255).astype(np.uint8)

            ax_res = axes[row_idx * 2 + 1, col]
            ax_res.imshow(residual)
            if col == 0:
                ax_res.set_ylabel(f'{method_labels[row_idx]}\n(×10 residual)', fontsize=8)
            ax_res.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.suptitle('Figure 5: Visual Quality Comparison + Residual (×10)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig5_visual_quality.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved → {out_dir}/fig5_visual_quality.png]")


PAYLOAD_CONFIGS = [
    (256, 8, False),
    (512, 4, True),
    (1024, 4, False),
    (2048, 2, False),
]


def run_fig6_payload_visual(N=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, prompt_key = get_dataset()
    out_dir = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    prompts = [dataset[i][prompt_key] for i in range(N)]

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available")
        return

    col_labels = [f'{p}b (hw={h})' for p, h, _ in PAYLOAD_CONFIGS]

    fig, axes = plt.subplots(N, len(PAYLOAD_CONFIGS), figsize=(5 * len(PAYLOAD_CONFIGS), 5 * N))
    if N == 1:
        axes = axes.reshape(1, -1)
    for row, prompt in enumerate(prompts):
        for col, (payload, hw, ecc) in enumerate(PAYLOAD_CONFIGS):
            pipe = load_pipe(device=device, method='fdw')
            set_random_seed(row)
            wm = FDW_Watermark(ch_factor=1, hw_factor=hw, fpr=1e-6, user_number=1_000_000,
                               payload_bits=payload, use_ecc=ecc, lambda_freq=0.02,
                               alpha_max=0.0, t_start=0.2, t_end=0.6,
                               use_fd_detect=False, use_spatial_perm=False)
            init_w = wm.create_watermark_and_return_w()
            fdsc_fft = wm.get_fdsc_template()
            out = pipe(prompt, num_images_per_prompt=1, guidance_scale=7.5,
                       num_inference_steps=50, height=512, width=512, latents=init_w,
                       fdsc_template_fft=fdsc_fft, fdsc_t_start=0.2, fdsc_t_end=0.6,
                       fdsc_alpha_max=0.0, template_mask=None, template_pattern=None,
                       template_t_start=0.3, template_t_end=1.0,
                       template_gamma=4.0, template_mode='shallow')
            axes[row, col].imshow(out.images[0])
            if row == 0:
                axes[row, col].set_title(col_labels[col], fontsize=10)
            axes[row, col].axis('off')
            del pipe
            torch.cuda.empty_cache()

    plt.suptitle('Figure 6: FDW Visual Effect at Different Payload Sizes', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig6_payload_visual.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved → {out_dir}/fig6_payload_visual.png]")


def run_quality(N):
    import open_clip
    from tqdm import tqdm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, prompt_key = get_dataset()
    out_dir = os.path.join(OUTPUT_DIR, 'quality')
    os.makedirs(out_dir, exist_ok=True)
    lf = _open_log(out_dir, 'quality')

    ref_model, _, ref_preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', device=device)
    ref_tokenizer = open_clip.get_tokenizer('ViT-L-14')

    quality_results = {}
    n_groups = min(10, N)

    for method in ['fdw', 'gs', 'dwdct', 'clean']:
        _log(f"\n[Quality] {method.upper()}")
        group_scores = []
        for g in range(n_groups):
            start = g * (N // n_groups)
            count = N // n_groups
            pipe = load_pipe(device=device, method=(method if method not in ('clean', 'dwdct') else 'fdw'))
            text_emb = pipe.get_text_embedding('')
            scores = []
            for i in tqdm(range(start, start + count), desc=f'  Group {g}'):
                prompt = dataset[i % len(dataset)][prompt_key]
                set_random_seed(i)
                if method == 'clean':
                    z = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
                    out = pipe(prompt, num_images_per_prompt=1, guidance_scale=7.5,
                               num_inference_steps=50, height=512, width=512, latents=z)
                    sim = measure_similarity([out.images[0]], prompt, ref_model,
                                             ref_preprocess, ref_tokenizer, device)
                    scores.append(sim[0].item())
                elif method == 'dwdct':
                    z = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
                    out = pipe(prompt, num_images_per_prompt=1, guidance_scale=7.5,
                               num_inference_steps=50, height=512, width=512, latents=z)
                    dwdct = DwtDctWatermark()
                    wm_img = dwdct.embed(out.images[0])
                    sim = measure_similarity([wm_img], prompt, ref_model,
                                             ref_preprocess, ref_tokenizer, device)
                    scores.append(sim[0].item())
                elif method == 'fdw':
                    wm = make_fdw_wm()
                    init_w = wm.create_watermark_and_return_w()
                    fdsc_fft = wm.get_fdsc_template()
                    tmask, tpat = wm.get_x_template()
                    out = pipe(prompt, num_images_per_prompt=1, guidance_scale=7.5,
                               num_inference_steps=50, height=512, width=512, latents=init_w,
                               fdsc_template_fft=fdsc_fft, fdsc_t_start=0.2, fdsc_t_end=0.6,
                               fdsc_alpha_max=0.0, template_mask=tmask, template_pattern=tpat,
                               template_t_start=0.3, template_t_end=1.0,
                               template_gamma=4.0, template_mode='shallow')
                    sim = measure_similarity([out.images[0]], prompt, ref_model,
                                             ref_preprocess, ref_tokenizer, device)
                    scores.append(sim[0].item())
                else:
                    wm = make_gs_wm()
                    init_w = wm.create_watermark_and_return_w()
                    out = pipe(prompt, num_images_per_prompt=1, guidance_scale=7.5,
                               num_inference_steps=50, height=512, width=512, latents=init_w)
                    sim = measure_similarity([out.images[0]], prompt, ref_model,
                                             ref_preprocess, ref_tokenizer, device)
                    scores.append(sim[0].item())
            group_scores.append(mean(scores))
            del pipe
            torch.cuda.empty_cache()

        quality_results[method] = {
            'mean': mean(group_scores),
            'std': stdev(group_scores) if len(group_scores) > 1 else 0.0,
            'groups': group_scores,
        }

    from scipy.stats import ttest_ind
    _log('\n' + '=' * 60)
    _log('Table 5: Image Quality (CLIP Score)')
    _log('=' * 60)
    clean_scores = quality_results.get('clean', {}).get('groups', [])
    for method in ['fdw', 'gs', 'dwdct', 'clean']:
        r = quality_results[method]
        if method != 'clean' and clean_scores:
            t_val, p_val = ttest_ind(clean_scores, r['groups'])
            _log(f"  {method:12s}: {r['mean']:.4f} ± {r['std']:.4f}  (t={t_val:.3f}, p={p_val:.4f})")
        else:
            _log(f"  {method:12s}: {r['mean']:.4f} ± {r['std']:.4f}  (baseline)")

    with open(os.path.join(out_dir, 'quality_results.json'), 'w') as f:
        json.dump(quality_results, f, indent=2)
    _close_log(lf)
    _log(f"\n[Saved → {out_dir}/]")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='FDW Complete Experiment Suite')
    parser.add_argument('experiment', choices=['baseline', 'ablation', 'sweep',
                                                'geometric', 'quality',
                                                'lambda_sweep', 'gamma_sweep', 'tstar_sweep',
                                                'capacity_sweep',
                                                'fig1', 'fig5', 'fig6',
                                                'all'],
                        help='Which experiment to run')
    parser.add_argument('--N', type=int, default=1000, help='Number of images per method')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"FDW Experiment Suite — N={args.N}")
    print(f"Output: {OUTPUT_DIR}/\n")

    dispatch = {
        'baseline':        lambda: run_baseline(args.N),
        'ablation':        lambda: run_ablation(args.N),
        'sweep':           lambda: run_sweep(args.N),
        'geometric':       lambda: run_geometric(args.N),
        'quality':         lambda: run_quality(args.N),
        'lambda_sweep':    lambda: run_lambda_sweep(args.N),
        'gamma_sweep':     lambda: run_gamma_sweep(args.N),
        'tstar_sweep':     lambda: run_tstar_sweep(args.N),
        'capacity_sweep':  lambda: run_capacity_sweep(args.N),
        'fig1':            lambda: run_fig1_attack_examples(),
        'fig5':            lambda: run_fig5_visual_quality(N=3),
        'fig6':            lambda: run_fig6_payload_visual(N=3),
    }

    if args.experiment == 'all':
        run_baseline(args.N)
        run_capacity_sweep(args.N)
        run_ablation(args.N)
        run_sweep(args.N)
        run_geometric(args.N)
        run_lambda_sweep(args.N)
        run_gamma_sweep(args.N)
        run_tstar_sweep(args.N)
        run_quality(args.N)
        run_fig1_attack_examples()
        run_fig5_visual_quality(N=3)
        run_fig6_payload_visual(N=3)
    else:
        dispatch[args.experiment]()


if __name__ == '__main__':
    main()
