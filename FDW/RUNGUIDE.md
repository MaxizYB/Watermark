# FDW 运行指南

## 目录结构

```
v3/
├── Gaussian-Shading/          # 原始 GS 代码（不修改）
└── FDW/                       # 本项目
    ├── watermark_fdw.py        # FDW 核心水印类（频域增强 + BCH + 双路检测）
    ├── fdw_pipeline.py         # 带 FDSC 钩子的 SD Pipeline
    ├── attacks.py              # 完整攻击库（30+ 攻击）
    ├── run_fdw.py              # 单次实验 + FDW vs GS 对比
    ├── run_attack_benchmark.py # 全攻击 benchmark
    └── scripts/
        ├── smoke_test.sh       # 快速冒烟测试（10张图）
        ├── run_benchmark.sh    # 完整 benchmark（100张图）
        ├── run_ablation.sh     # 消融实验
        └── run_hyperparam_sweep.sh  # 超参数搜索
```

---

## 环境安装

```bash
# 1. 进入 GS 目录安装基础依赖
cd v3/Gaussian-Shading
pip install -r requirements.txt

# 2. 额外依赖
pip install bchlib          # BCH 纠错码（可选，无则自动用重复码 fallback）
pip install matplotlib      # 绘图（可选）
pip install pycryptodome    # ChaCha20（GS 已有）

# 3. 确认 diffusers 版本
pip install diffusers==0.21.4 transformers accelerate
```

---

## 快速开始

所有命令在 `v3/FDW/` 目录下运行。

### 1. 冒烟测试（验证环境）

```bash
cd v3/FDW
bash scripts/smoke_test.sh
```

10 张图，clean + jpeg_50，FDW 和 GS 各跑一遍，约 5 分钟（A100）。

### 2. 单次实验

```bash
# FDW，clean，100 张
python run_fdw.py --method fdw --num 100 --attack clean

# GS baseline，jpeg_50
python run_fdw.py --method gs --num 100 --attack jpeg_50

# FDW vs GS 对比，一次输出对比表
python run_fdw.py --method both --num 1 --attack jpeg_50

python run_fdw.py --method both --num 1 --attack rotate_15 \
  --reference_model ViT-B-32 \
  --reference_model_pretrain openai

python run_fdw.py --method both --num 1 --attack rotate_15
```




输出示例：
```
======================================================================
  Metric                    FDW      GS Baseline         Delta
----------------------------------------------------------------------
  TPR Detection          0.9200          0.8100        +0.1100
  TPR Traceability       0.8900          0.7600        +0.1300
  Mean Acc               0.9450          0.8820        +0.0630
  CLIP Score             0.0000          0.0000        +0.0000
======================================================================
```

### 3. 全攻击 Benchmark

```bash
bash scripts/run_benchmark.sh
# 或手动：
python run_attack_benchmark.py --method both --num 1 --plot
```

输出：
- `benchmark_output/benchmark_results.json`：完整数值结果
- `benchmark_output/benchmark_plot.png`：TPR 和 Acc 折线图

### 4. 消融实验

```bash
bash scripts/run_ablation.sh
```

对比四个配置：
| 配置 | 频域初始噪声 | ECC | FDSC | 双路检测 |
|------|:-----------:|:---:|:----:|:-------:|
| GS baseline | ✗ | ✗ | ✗ | ✗ |
| fdw_freq_only | ✓ | ✗ | ✗ | ✗ |
| fdw_freq_ecc | ✓ | ✓ | ✗ | ✗ |
| fdw_full | ✓ | ✓ | ✓ | ✓ |

### 5. 超参数搜索

```bash
bash scripts/run_hyperparam_sweep.sh
```

扫描 `lambda_freq ∈ {0.04, 0.08, 0.12, 0.16}` × `alpha_max ∈ {0.005, 0.010, 0.015, 0.020}`，共 16 组。

---

## 主要参数说明

### run_fdw.py / run_attack_benchmark.py 共用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--method` | `fdw` | `fdw` / `gs` / `both` |
| `--num` | `100` | 测试图像数量 |
| `--model_path` | `stabilityai/stable-diffusion-2-1-base` | SD 模型路径（本地或 HF） |
| `--dataset_path` | `Gustavosta/Stable-Diffusion-Prompts` | Prompt 数据集 |
| `--attack` | `clean` | 攻击名称（见下表） |
| `--num_inference_steps` | `50` | 生成步数 |
| `--num_inversion_steps` | 同上 | DDIM Inversion 步数 |
| `--channel_copy` | `1` | GS 通道冗余因子 |
| `--hw_copy` | `1` | GS 空间冗余因子 |

### FDW 专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--payload_bits` | `48` | 有效载荷比特数（ECC 前） |
| `--use_ecc` / `--no_ecc` | 开启 | BCH 纠错编码 |
| `--lambda_freq` | `0.08` | 初始噪声频域叠加强度 |
| `--use_fdsc` / `--no_fdsc` | 开启 | 扩散过程中的 FDSC 约束 |
| `--alpha_max` | `0.015` | FDSC 最大扰动强度 |
| `--fdsc_t_start` | `0.2` | FDSC 开始时间步比例 |
| `--fdsc_t_end` | `0.6` | FDSC 结束时间步比例 |
| `--use_fd_detect` / `--no_fd_detect` | 开启 | 双路（空间+频域）检测 |

---

## 攻击名称速查

### 单次实验 `--attack` 可用值

| 类别 | 攻击名 | 说明 |
|------|--------|------|
| Clean | `clean` | 无攻击 |
| 几何 | `rotate_15`, `rotate_45` | 旋转 |
| | `scale_075`, `scale_050` | 缩放 |
| | `crop_080`, `crop_060` | 随机裁剪 |
| | `translate`, `flip_h` | 平移、翻转 |
| 光度 | `brightness_2`, `brightness_6` | 亮度 |
| | `contrast_2`, `saturation_2` | 对比度、饱和度 |
| | `color_jitter` | 综合颜色抖动 |
| 降质 | `jpeg_75`, `jpeg_50`, `jpeg_25` | JPEG 压缩 |
| | `gauss_blur_2`, `gauss_blur_4` | 高斯模糊 |
| | `median_blur_7` | 中值滤波 |
| | `gauss_noise_005`, `gauss_noise_010` | 高斯噪声 |
| | `sp_noise_005` | 椒盐噪声 |
| | `resize_050`, `resize_025` | 缩放重采样 |
| | `pixelate_16` | 像素化 |
| 对抗 | `adversarial_8`, `adversarial_16` | 对抗扰动 |
| Stirmark | `stirmark_rst` | 旋转+缩放+平移 |
| | `stirmark_all` | 综合 Stirmark |
| 再生成 | `regeneration` | VAE 重编解码（需传 pipe） |

### Benchmark `--attack_groups` 可用值

`clean` / `geometric` / `photometric` / `degradation` / `adversarial` / `stirmark`

---

## 输出文件说明

```
output/
├── fdw/
│   ├── results_clean.json       # TPR, Acc, CLIP Score
│   └── images/                  # 生成图像（--save_images 时）
└── gs/
    └── results_clean.json

benchmark_output/
├── benchmark_results.json       # 所有攻击 × 所有方法的完整结果
└── benchmark_plot.png           # 可视化折线图
```

`results_*.json` 格式：
```json
{
  "method": "fdw",
  "tpr_detection": 0.92,
  "tpr_traceability": 0.89,
  "mean_acc": 0.945,
  "std_acc": 0.023,
  "mean_clip": 0.0,
  "std_clip": 0.0
}
```

---

## 常见问题

**Q: 模型下载很慢**
```bash
# 提前下载到本地
huggingface-cli download stabilityai/stable-diffusion-2-1-base --local-dir ./models/sd21
python run_fdw.py --model_path ./models/sd21 ...
```

**Q: bchlib 安装失败**
```bash
pip install bchlib
# 如果失败，代码会自动 fallback 到 3× 重复码，功能不受影响
```

**Q: CUDA OOM**
- 减少 `--num_inference_steps`（如 20）
- 减少 `--num`（先用 10 测试）
- 使用 `--image_length 256`

**Q: 想只测试 FDW 的频域初始噪声效果（不用 FDSC）**
```bash
python run_fdw.py --method fdw --no_fdsc --attack jpeg_50 --num 50
```

**Q: 想复现 GS 原始结果**
```bash
python run_fdw.py --method gs --channel_copy 1 --hw_copy 1 --attack clean --num 100
# 等价于 GS 原始 run_gaussian_shading.py
```
