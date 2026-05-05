"""
Attack benchmark: sweep all attacks for both FDW and GS baseline,
produce a summary table and optionally a matplotlib plot.

Usage:
    python run_attack_benchmark.py --method both --num 50
    python run_attack_benchmark.py --method fdw  --num 100 --attack_groups geometric degradation
"""

import argparse
import os
import sys
import json
from tqdm import tqdm
from statistics import mean, stdev

import torch
import numpy as np

GS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Gaussian-Shading')
sys.path.insert(0, GS_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from diffusers import DDIMScheduler
from image_utils import set_random_seed, transform_img
from watermark import Gaussian_Shading, Gaussian_Shading_chacha
from fdw_pipeline import FDWStableDiffusionPipeline, InversableStableDiffusionPipeline


def get_dataset(args):
    from datasets import load_dataset
    if 'coco' in args.dataset_path.lower():
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)['annotations']
        return dataset, 'caption'
    else:
        dataset = load_dataset(args.dataset_path)['train']
        sample = dataset[0]
        for key in ('Prompt', 'TEXT', 'text', 'caption', 'prompt'):
            if key in sample:
                return dataset, key
        raise ValueError(f"Cannot detect prompt key. Keys: {list(sample.keys())}")

from watermark_fdw import FDW_Watermark, blind_detect_rotation
from fdw_pipeline import FDWStableDiffusionPipeline
from attacks import get_attack, ATTACK_GROUPS, ATTACK_REGISTRY


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pipe(model_path, method, device):
    sched = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
    cls = FDWStableDiffusionPipeline if method == 'fdw' else InversableStableDiffusionPipeline
    pipe = cls.from_pretrained(model_path, scheduler=sched, torch_dtype=torch.float16)
    pipe.safety_checker = None
    return pipe.to(device)


def make_wm(args, method):
    if method == 'fdw':
        return FDW_Watermark(
            ch_factor=args.channel_copy, hw_factor=args.hw_copy,
            fpr=args.fpr, user_number=args.user_number,
            payload_bits=args.payload_bits, use_ecc=args.use_ecc,
            lambda_freq=args.lambda_freq, alpha_max=args.alpha_max,
            t_start=args.fdsc_t_start, t_end=args.fdsc_t_end,
            use_fd_detect=args.use_fd_detect,
        )
    elif args.chacha:
        return Gaussian_Shading_chacha(args.channel_copy, args.hw_copy,
                                       args.fpr, args.user_number)
    else:
        return Gaussian_Shading(args.channel_copy, args.hw_copy,
                                args.fpr, args.user_number)


def generate_images(args, method, pipe, dataset, prompt_key, device):
    """Pre-generate all watermarked images. Each image gets its own watermark instance."""
    text_emb = pipe.get_text_embedding('')
    images = []
    watermarks = []

    for i in tqdm(range(args.num), desc=f"[{method.upper()}] Generating"):
        seed = i + args.gen_seed
        set_random_seed(seed)

        # Fresh watermark object per image so state doesn't get overwritten
        wm = make_wm(args, method)
        init_w = wm.create_watermark_and_return_w()

        if method == 'fdw':
            fdsc_fft = wm.get_fdsc_template() if args.use_fdsc else None
            template_mask, template_pattern = wm.get_x_template() if args.use_template_injection else (None, None)
            out = pipe(
                dataset[i][prompt_key],
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length, width=args.image_length,
                latents=init_w,
                fdsc_template_fft=fdsc_fft,
                fdsc_t_start=args.fdsc_t_start,
                fdsc_t_end=args.fdsc_t_end,
                fdsc_alpha_max=args.alpha_max,
                template_mask=template_mask,
                template_pattern=template_pattern,
                template_t_start=args.template_t_start,
                template_t_end=args.template_t_end,
                template_gamma=args.template_gamma,
                template_mode=args.template_mode,
            )
        else:
            out = pipe(
                dataset[i][prompt_key],
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length, width=args.image_length,
                latents=init_w,
            )

        images.append(out.images[0])
        watermarks.append(wm)

    return images, watermarks, text_emb


def eval_attack(attack_name, images, watermarks, pipe, text_emb, args, device):
    """Evaluate one attack across all pre-generated images."""
    attack_fn = get_attack(attack_name, pipe if attack_name == 'regeneration' else None)
    acc_list = []

    for img, wm in zip(images, watermarks):
        attacked = attack_fn(img)

        # Blind geometric correction: scale (black border) + rotation (FFT template)
        if args.geo_correct and isinstance(wm, FDW_Watermark) and args.use_template_injection:
            from attacks import detect_and_correct_geom
            img_t_tensor = transform_img(attacked).unsqueeze(0).to(text_emb.dtype).to(device)
            latents_raw = pipe.get_image_latents(img_t_tensor, sample=False)
            zT_raw = pipe.forward_diffusion(
                latents=latents_raw, text_embeddings=text_emb,
                guidance_scale=1, num_inference_steps=args.num_inversion_steps,
            )
            corrected_img, _, _, _ = detect_and_correct_geom(zT_raw, attacked, args.image_length)
            img_t = transform_img(corrected_img).unsqueeze(0).to(text_emb.dtype).to(device)
        else:
            img_t = transform_img(attacked).unsqueeze(0).to(text_emb.dtype).to(device)

        latents = pipe.get_image_latents(img_t, sample=False)
        rev = pipe.forward_diffusion(
            latents=latents, text_embeddings=text_emb,
            guidance_scale=1, num_inference_steps=args.num_inversion_steps,
        )

        acc = wm.eval_watermark(rev)
        acc_list.append(acc)

    # Aggregate TPR counts across all per-image watermark objects
    tpr_det   = sum(wm.tp_onebit_count for wm in watermarks)
    tpr_trace = sum(wm.tp_bits_count   for wm in watermarks)
    for wm in watermarks:
        wm.tp_onebit_count = 0
        wm.tp_bits_count   = 0

    return {
        'tpr_detection':    tpr_det   / args.num,
        'tpr_traceability': tpr_trace / args.num,
        'mean_acc': mean(acc_list),
        'std_acc':  stdev(acc_list) if len(acc_list) > 1 else 0.0,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(all_results: dict, output_path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not available, skipping plot.")
        return

    attacks = list(next(iter(all_results.values())).keys())
    methods = list(all_results.keys())
    colors = {'fdw': '#7B2D8B', 'gs': '#C8A951'}
    markers = {'fdw': 'o', 'gs': 's'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for metric, ax, ylabel in [
        ('tpr_detection', axes[0], 'TPR (Detection)'),
        ('mean_acc',      axes[1], 'Mean Bit Accuracy'),
    ]:
        x = np.arange(len(attacks))
        for method in methods:
            vals = [all_results[method][atk][metric] for atk in attacks]
            ax.plot(x, vals, marker=markers.get(method, 'o'),
                    color=colors.get(method, 'gray'),
                    label=method.upper(), linewidth=2, markersize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(ylabel)

    plt.tight_layout()
    plot_path = os.path.join(output_path, 'benchmark_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[Plot saved → {plot_path}]")
    plt.close()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(all_results: dict):
    methods = list(all_results.keys())
    attacks = list(next(iter(all_results.values())).keys())

    col_w = 14
    header = f"{'Attack':<22}" + "".join(
        f"{'TPR_' + m.upper():>{col_w}}{'ACC_' + m.upper():>{col_w}}"
        for m in methods
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for atk in attacks:
        row = f"{atk:<22}"
        for m in methods:
            r = all_results[m][atk]
            row += f"{r['tpr_detection']:>{col_w}.4f}{r['mean_acc']:>{col_w}.4f}"
        print(row)
    print("=" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    # Dch attacks to run
    if args.attack_groups:
        attack_names = []
        for g in args.attack_groups:
            attack_names.extend(ATTACK_GROUPS.get(g, []))
    elif args.attacks:
        attack_names = args.attacks
    else:
        # Default: representative subset
        attack_names = [
            'clean',
            'jpeg_75', 'jpeg_50', 'jpeg_25',
            'gauss_blur_4', 'gauss_noise_005',
            'crop_080', 'crop_060',
            'rotate_15', 'rotate_45',
            'scale_075', 'resize_025',
            'brightness_2', 'color_jitter',
            'adversarial_8',
            'stirmark_rst', 'stirmark_all',
        ]

    methods = ['fdw', 'gs'] if args.method == 'both' else [args.method]
    dataset, prompt_key = get_dataset(args)
    os.makedirs(args.output_path, exist_ok=True)

    all_results = {}

    for method in methods:
        print(f"\n{'='*60}\nPre-generating images: {method.upper()}\n{'='*60}")
        pipe = load_pipe(args.model_path, method, device)
        images, watermarks, text_emb = generate_images(
            args, method, pipe, dataset, prompt_key, device)

        method_results = {}
        for atk in tqdm(attack_names, desc=f"[{method.upper()}] Attacks"):
            print(f"  → {atk}")
            res = eval_attack(atk, images, watermarks, pipe, text_emb, args, device)
            method_results[atk] = res
            print(f"     TPR={res['tpr_detection']:.4f}  Acc={res['mean_acc']:.4f}")

        all_results[method] = method_results
        del pipe
        torch.cuda.empty_cache()

    # Save JSON
    out_file = os.path.join(args.output_path, 'benchmark_results.json')
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Full results saved → {out_file}]")

    # Print table
    print_summary_table(all_results)

    # Plot
    if args.plot:
        plot_results(all_results, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FDW Attack Benchmark')

    parser.add_argument('--method', default='both', choices=['fdw', 'gs', 'both'])
    parser.add_argument('--model_path', default='Manojb/stable-diffusion-2-1-base')
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--num', default=50, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)

    # Watermark
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=4, type=int)
    parser.add_argument('--fpr', default=1e-6, type=float)
    parser.add_argument('--user_number', default=1_000_000, type=int)
    parser.add_argument('--chacha', action='store_true')

    # FDW
    parser.add_argument('--payload_bits', default=512, type=int)
    parser.add_argument('--use_ecc', action='store_true', default=True)
    parser.add_argument('--no_ecc', dest='use_ecc', action='store_false')
    parser.add_argument('--lambda_freq', default=0.08, type=float)
    parser.add_argument('--use_fdsc', action='store_true', default=True)
    parser.add_argument('--no_fdsc', dest='use_fdsc', action='store_false')
    parser.add_argument('--alpha_max', default=0.015, type=float)
    parser.add_argument('--fdsc_t_start', default=0.2, type=float)
    parser.add_argument('--fdsc_t_end', default=0.6, type=float)
    parser.add_argument('--use_fd_detect', action='store_true', default=False)
    parser.add_argument('--no_fd_detect', dest='use_fd_detect', action='store_false')
    parser.add_argument('--use_template_injection', action='store_true', default=True)
    parser.add_argument('--no_template_injection', dest='use_template_injection', action='store_false')
    parser.add_argument('--template_t_start', default=0.3, type=float,
                        help='Shallow mode: injection timestep t* (e.g. 0.3). Maxsive mode: start of window.')
    parser.add_argument('--template_t_end', default=1.0, type=float,
                        help='End of injection window (only used in maxsive mode)')
    parser.add_argument('--template_gamma', default=8.0, type=float)
    parser.add_argument('--template_mode', default='shallow', choices=['shallow', 'maxsive'],
                        help='Template injection mode: shallow (ShallowDiffuse, single step at t*) or maxsive (all steps)')
    parser.add_argument('--use_pixel_template', action='store_true', default=False)
    parser.add_argument('--no_pixel_template', dest='use_pixel_template', action='store_false')
    parser.add_argument('--pixel_template_gamma', default=50.0, type=float)
    parser.add_argument('--geo_search_steps', default=10, type=int,
                        help='DDIM steps for brute-force rotation search')
    parser.add_argument('--geo_correct', action='store_true', default=True)
    parser.add_argument('--no_geo_correct', dest='geo_correct', action='store_false')

    # Attack selection
    parser.add_argument('--attack_groups', nargs='+', default=None,
                        choices=list(ATTACK_GROUPS.keys()),
                        help='Attack groups to run')
    parser.add_argument('--attacks', nargs='+', default=None,
                        choices=list(ATTACK_REGISTRY.keys()) + ['regeneration'],
                        help='Specific attacks to run')

    # Output
    parser.add_argument('--output_path', default='./benchmark_output/')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no_plot', dest='plot', action='store_false')

    args = parser.parse_args()
    main(args)
