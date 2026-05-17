"""
Main experiment runner: FDW vs Gaussian Shading baseline comparison.

Generates watermarked images, applies a single distortion, evaluates TPR/BER,
and optionally computes CLIP Score and FID.

Usage:
    python run_fdw.py --method fdw --num 100
    python run_fdw.py --method gs  --num 100
    python run_fdw.py --method both --num 100 --attack jpeg_50
"""

import argparse
import os
import sys
import json
import copy
from tqdm import tqdm
from statistics import mean, stdev

import torch
import numpy as np
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
GS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Gaussian-Shading')
sys.path.insert(0, GS_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import open_clip

from image_utils import set_random_seed, transform_img, measure_similarity
from watermark import Gaussian_Shading, Gaussian_Shading_chacha
# Import via fdw_pipeline so the diffusers-0.37 compatibility patch is applied
from fdw_pipeline import FDWStableDiffusionPipeline, InversableStableDiffusionPipeline


def get_dataset(args):
    """Load prompt dataset. Supports HuggingFace datasets or local COCO JSON."""
    from datasets import load_dataset
    if 'coco' in args.dataset_path.lower():
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)['annotations']
        return dataset, 'caption'
    else:
        dataset = load_dataset(args.dataset_path)['train']
        # Detect prompt key
        sample = dataset[0]
        for key in ('Prompt', 'TEXT', 'text', 'caption', 'prompt'):
            if key in sample:
                return dataset, key
        raise ValueError(f"Cannot detect prompt key in dataset. Keys: {list(sample.keys())}")

from watermark_fdw import FDW_Watermark, blind_detect_rotation
from attacks import get_attack, ATTACK_REGISTRY


# ── Pipeline loader ───────────────────────────────────────────────────────────

def load_pipeline(model_path: str, method: str, device: str):
    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
    if method == 'fdw':
        pipe = FDWStableDiffusionPipeline.from_pretrained(
            model_path, scheduler=scheduler,
            torch_dtype=torch.float16,
        )
    else:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            model_path, scheduler=scheduler,
            torch_dtype=torch.float16,
        )
    pipe.safety_checker = None
    return pipe.to(device)


# ── Single-method evaluation ──────────────────────────────────────────────────

def run_method(args, method: str, pipe, attack_fn, ref_model=None,
               ref_clip_preprocess=None, ref_tokenizer=None):
    device = pipe.device

    # Watermark object
    if method == 'fdw':
        wm = FDW_Watermark(
            ch_factor=args.channel_copy,
            hw_factor=args.hw_copy,
            fpr=args.fpr,
            user_number=args.user_number,
            payload_bits=args.payload_bits,
            use_ecc=args.use_ecc,
            lambda_freq=args.lambda_freq,
            alpha_max=args.alpha_max,
            t_start=args.fdsc_t_start,
            t_end=args.fdsc_t_end,
            use_fd_detect=args.use_fd_detect,
        )
    elif args.chacha:
        wm = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy,
                                     args.fpr, args.user_number)
    else:
        wm = Gaussian_Shading(args.channel_copy, args.hw_copy,
                              args.fpr, args.user_number)

    dataset, prompt_key = get_dataset(args)
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    acc_list = []
    clip_scores = []

    for i in tqdm(range(args.num), desc=f"[{method.upper()}]"):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        set_random_seed(seed)
        init_latents_w = wm.create_watermark_and_return_w()

        # Generate watermarked image
        if method == 'fdw':
            fdsc_fft = wm.get_fdsc_template() if args.use_fdsc else None
            template_mask, template_pattern = wm.get_x_template() if args.use_template_injection else (None, None)
            outputs = pipe(
                current_prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_w,
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
            outputs = pipe(
                current_prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_w,
            )

        image_w = outputs.images[0]

        # Save image if requested
        if args.save_images:
            os.makedirs(os.path.join(args.output_path, method, 'images'), exist_ok=True)
            image_w.save(os.path.join(args.output_path, method, 'images', f'{i:04d}.png'))

        # Apply attack
        image_attacked = attack_fn(image_w)

        # Blind geometric correction: preprocess (scale+translate) → rotation
        if method == 'fdw' and args.geo_correct and args.use_template_injection:
            from attacks import detect_and_correct_geom
            img_tensor = transform_img(image_attacked).unsqueeze(0).to(text_embeddings.dtype).to(device)
            raw_latents = pipe.get_image_latents(img_tensor, sample=False)
            zT_raw = pipe.forward_diffusion(
                latents=raw_latents, text_embeddings=text_embeddings,
                guidance_scale=1, num_inference_steps=args.num_inversion_steps,
            )
            corrected_img, _, _, _ = detect_and_correct_geom(zT_raw, image_attacked, args.image_length)
            img_tensor = transform_img(corrected_img).unsqueeze(0).to(text_embeddings.dtype).to(device)
        else:
            img_tensor = transform_img(image_attacked).unsqueeze(0).to(text_embeddings.dtype).to(device)

        # DDIM inversion
        image_latents = pipe.get_image_latents(img_tensor, sample=False)
        reversed_latents = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inversion_steps,
        )

        # Evaluate
        acc = wm.eval_watermark(reversed_latents)
        acc_list.append(acc)

        # CLIP Score
        if ref_model is not None:
            score = measure_similarity([image_w], current_prompt, ref_model,
                                       ref_clip_preprocess, ref_tokenizer, device)
            clip_scores.append(score[0].item())
        else:
            clip_scores.append(0.0)

    tpr_det, tpr_trace = wm.get_tpr()
    return {
        'method': method,
        'tpr_detection': tpr_det / args.num,
        'tpr_traceability': tpr_trace / args.num,
        'mean_acc': mean(acc_list),
        'std_acc': stdev(acc_list) if len(acc_list) > 1 else 0.0,
        'mean_clip': mean(clip_scores),
        'std_clip': stdev(clip_scores) if len(clip_scores) > 1 else 0.0,
        'acc_list': acc_list,
    }


# ── Results saving ────────────────────────────────────────────────────────────

def save_results(results: dict, output_path: str, attack_name: str):
    os.makedirs(output_path, exist_ok=True)
    fname = os.path.join(output_path, f'results_{attack_name}.json')
    # Remove non-serialisable acc_list for JSON
    out = {k: v for k, v in results.items() if k != 'acc_list'}
    with open(fname, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[Results saved → {fname}]")
    _print_table(results)


def _print_table(results: dict):
    print("\n" + "=" * 60)
    print(f"  Method          : {results['method'].upper()}")
    print(f"  TPR (detection) : {results['tpr_detection']:.4f}")
    print(f"  TPR (traceab.)  : {results['tpr_traceability']:.4f}")
    print(f"  Mean Acc (BER)  : {results['mean_acc']:.4f} ± {results['std_acc']:.4f}")
    if results['mean_clip'] > 0:
        print(f"  CLIP Score      : {results['mean_clip']:.4f} ± {results['std_clip']:.4f}")
    print("=" * 60)


def compare_and_print(res_fdw: dict, res_gs: dict):
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {'FDW':>15} {'GS Baseline':>15} {'Delta':>12}")
    print("-" * 70)
    for key, label in [
        ('tpr_detection',    'TPR Detection'),
        ('tpr_traceability', 'TPR Traceability'),
        ('mean_acc',         'Mean Acc'),
        ('mean_clip',        'CLIP Score'),
    ]:
        v_fdw = res_fdw[key]
        v_gs  = res_gs[key]
        delta = v_fdw - v_gs
        sign  = '+' if delta >= 0 else ''
        print(f"  {label:<23} {v_fdw:>15.4f} {v_gs:>15.4f} {sign}{delta:>11.4f}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    attack_fn = get_attack(args.attack)

    # CLIP reference model
    ref_model = ref_clip_preprocess = ref_tokenizer = None
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    methods = ['fdw', 'gs'] if args.method == 'both' else [args.method]
    all_results = {}

    for method in methods:
        print(f"\n{'='*60}\nRunning method: {method.upper()}\n{'='*60}")
        pipe = load_pipeline(args.model_path, method, device)
        res = run_method(args, method, pipe, attack_fn,
                         ref_model, ref_clip_preprocess, ref_tokenizer)
        all_results[method] = res
        save_results(res, os.path.join(args.output_path, method), args.attack)
        del pipe
        torch.cuda.empty_cache()

    if args.method == 'both' and 'fdw' in all_results and 'gs' in all_results:
        compare_and_print(all_results['fdw'], all_results['gs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FDW vs GS Watermark Comparison')

    # Method
    parser.add_argument('--method', default='fdw', choices=['fdw', 'gs', 'both'],
                        help='Which method to run')

    # Dataset / model
    parser.add_argument('--model_path', default='Manojb/stable-diffusion-2-1-base')
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--num', default=100, type=int, help='Number of images')
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)

    # Watermark (shared)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=4, type=int)
    parser.add_argument('--fpr', default=1e-6, type=float)
    parser.add_argument('--user_number', default=1_000_000, type=int)
    parser.add_argument('--chacha', action='store_true')

    # FDW-specific
    parser.add_argument('--payload_bits', default=512, type=int)
    parser.add_argument('--use_ecc', action='store_true', default=True)
    parser.add_argument('--no_ecc', dest='use_ecc', action='store_false')
    parser.add_argument('--lambda_freq', default=0.08, type=float,
                        help='Frequency overlay strength in initial noise')
    parser.add_argument('--use_fdsc', action='store_true', default=True,
                        help='Enable FDSC during denoising')
    parser.add_argument('--no_fdsc', dest='use_fdsc', action='store_false')
    parser.add_argument('--alpha_max', default=0.015, type=float,
                        help='Max FDSC perturbation strength')
    parser.add_argument('--fdsc_t_start', default=0.2, type=float)
    parser.add_argument('--fdsc_t_end', default=0.6, type=float)
    parser.add_argument('--use_fd_detect', action='store_true', default=False,
                        help='Enable dual-path (spatial+freq) detection (experimental)')
    parser.add_argument('--no_fd_detect', dest='use_fd_detect', action='store_false')
    parser.add_argument('--use_template_injection', action='store_true', default=True,
                        help='Enable ShallowDiffuse/MaXsive-style template injection during denoising')
    parser.add_argument('--no_template_injection', dest='use_template_injection', action='store_false')
    parser.add_argument('--template_t_start', default=0.3, type=float,
                        help='Shallow mode: injection timestep t* (e.g. 0.3). Maxsive mode: start of window.')
    parser.add_argument('--template_t_end', default=1.0, type=float,
                        help='End of injection window (only used in maxsive mode)')
    parser.add_argument('--template_gamma', default=8.0, type=float)
    parser.add_argument('--template_mode', default='shallow', choices=['shallow', 'maxsive'],
                        help='Template injection mode: shallow (ShallowDiffuse, single step at t*) or maxsive (all steps)')

    # Pixel-space template for rotation detection
    parser.add_argument('--use_pixel_template', action='store_true', default=False,
                        help='Inject X-template in pixel-space FFT for rotation detection')
    parser.add_argument('--no_pixel_template', dest='use_pixel_template', action='store_false')
    parser.add_argument('--pixel_template_gamma', default=50.0, type=float,
                        help='Strength of pixel-space template injection')
    parser.add_argument('--geo_search_steps', default=10, type=int,
                        help='DDIM steps for brute-force rotation search')

    # Attack
    parser.add_argument('--attack', default='clean',
                        choices=list(ATTACK_REGISTRY.keys()) + ['regeneration'],
                        help='Attack to apply')

    parser.add_argument('--geo_correct', action='store_true', default=True,
                        help='Enable geometric correction before DDIM inversion for FDW')
    parser.add_argument('--no_geo_correct', dest='geo_correct', action='store_false')

    # Output
    parser.add_argument('--output_path', default='./output/')
    parser.add_argument('--save_images', action='store_true', default=False)

    # CLIP
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)

    args = parser.parse_args()
    main(args)
