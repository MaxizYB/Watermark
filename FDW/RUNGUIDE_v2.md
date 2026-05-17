# FDW 完整实验运行指南 v2

> 对应文档：`docs/实验设计文档.md`
> 脚本：`run_experiments.py`
> 所有实验结果同时输出到控制台和 `experiment_output/` 目录下的日志文件

---

## 一、快速开始

```bash
cd v3/FDW
conda activate graph

# 小规模验证（约 10 分钟）
python run_experiments.py baseline --N 2

# 正式实验：全部实验（N=1000，约 2 周）
python run_experiments.py all --N 1000
```

---

## 二、环境依赖

```bash
conda activate graph
pip install tqdm matplotlib scipy open_clip_torch datasets
```

已有依赖（无需额外安装）：torch, diffusers, transformers, PIL, numpy

---

## 三、实验命令一览

| 实验 | 命令 | 产出 | 对应论文 | 耗时(N=1000) |
|------|------|------|----------|:---:|
| 基线对比 | `python run_experiments.py baseline --N 1000` | 表1、表2、图3 | §4.3 | 2-3天 |
| 容量-冗余权衡 | `python run_experiments.py capacity_sweep --N 1000` | 表3 | §7.1 | 1-2天 |
| 组件消融 | `python run_experiments.py ablation --N 1000` | 表4 | §7.5 | 1-2天 |
| 攻击强度扫描 | `python run_experiments.py sweep --N 1000` | 图2 | §4.4 | 2天 |
| 几何攻击专项 | `python run_experiments.py geometric --N 1000` | 图4 | §4.4 | 1天 |
| λ 扫描 | `python run_experiments.py lambda_sweep --N 1000` | 表+图 | §7.2 | 1天 |
| γ 扫描 | `python run_experiments.py gamma_sweep --N 1000` | 表+图 | §7.3 | 1天 |
| t* 扫描 | `python run_experiments.py tstar_sweep --N 1000` | 表+图 | §7.4 | 1天 |
| 图像质量 | `python run_experiments.py quality --N 1000` | 表5 | §4.3 | 3-4天 |
| 攻击示例图 | `python run_experiments.py fig1` | 图1 | — | 5分钟 |
| 视觉质量图 | `python run_experiments.py fig5` | 图5 | — | 30分钟 |
| 载荷效果图 | `python run_experiments.py fig6` | 图6 | — | 30分钟 |
| **全部** | `python run_experiments.py all --N 1000` | 以上全部 | — | 10+天 |

---

## 四、各实验详细说明

### 4.1 基线对比 (`baseline`)

**对比方法：** FDW (512b) / Gaussian Shading (256b) / DwtDct (256b)

**测试攻击（17项）：** clean, jpeg_75, jpeg_50, jpeg_25, gauss_blur_4, gauss_noise_005, crop_080, crop_060, rotate_15, rotate_45, scale_075, resize_025, brightness_2, color_jitter, adversarial_8, stirmark_rst, stirmark_all

**产出文件：**
```
experiment_output/baseline/
├── baseline_results.json      # 完整数值（每方法 × 每攻击: TPR_det, TPR_trace, Acc, Std）
├── table2_per_attack.txt      # 逐攻击详细对比表 + Average 行
├── table1_summary.txt         # 聚合表（单行/方法: TPR_Clean / TPR_Adv / Acc_Clean / Acc_Adv / CLIP）
├── baseline.txt               # 运行日志（含表1、表2）
├── fig3_baseline.png          # 折线图（3方法 × 17攻击）
```

> 注：`table1_summary.txt` 需要先运行 `quality` 实验才会包含 CLIP-Score 列。

**控制台输出：** 表1（基线总表）+ 表2（逐攻击详细对比 + Average 行）

### 4.2 容量-冗余权衡 (`capacity_sweep`) — 表3

**4组配置，测试容量与冗余的权衡：**

| 配置 | hw_factor | 有效载荷 | 重复码 | FD-Init | 每 bit 投票 |
|------|-----------|---------|--------|---------|------------|
| hw8_256b | 8 | 256b | ✗ | ✗ | 64 |
| hw4_1024b | 4 | 1024b | ✗ | ✗ | 16 |
| hw4_512b_repeat | 4 | 512b | ✓ (2×) | ✗ | 32 |
| hw4_512b_repeat_fdinit | 4 | 512b | ✓ (2×) | ✓ | 32 |
| hw2_4096b | 2 | 4096b | ✗ | ✗ | 4 |

**测试攻击：** clean, jpeg_25, gauss_blur_4, crop_060, rotate_15, rotate_45

**产出文件：**
```
experiment_output/capacity_sweep/
├── capacity_sweep_results.json
├── capacity_sweep.txt          # 表3 + Average 行
```

### 4.3 组件消融 (`ablation`) — 表4

**6组配置逐步叠加组件：**

| 编号 | 配置 | FD-Init | X模板 | 几何校正 | 重复码 |
|:---:|------|:---:|:---:|:---:|:---:|
| 1 | GS baseline (hw=8, 256b) | ✗ | ✗ | ✗ | ✗ |
| 2 | Expand (hw=4, 1024b) | ✗ | ✗ | ✗ | ✗ |
| 3 | + FD-Init | ✓ | ✗ | ✗ | ✗ |
| 4 | + Repetition Code | ✓ | ✗ | ✗ | ✓ |
| 5 | + X-Template (no correct) | ✓ | ✓ | ✗ | ✓ |
| 6 | FDW Full | ✓ | ✓ | ✓ | ✓ |

**测试攻击：** clean, jpeg_25, gauss_blur_4, crop_060, rotate_15, rotate_45, scale_075

**产出文件：**
```
experiment_output/ablation/
├── ablation_results.json
├── ablation.txt
```

### 4.4 攻击强度扫描 (`sweep`) — 图2

**8种攻击 × 多个强度等级，FDW vs GS：**

| 攻击 | 扫描参数 |
|------|---------|
| JPEG Quality | 90, 75, 50, 35, 25, 15, 10 |
| Gaussian Blur | r = 2, 4, 6, 8, 10 |
| Gaussian Noise | σ = 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40 |
| Brightness | ×2, ×4, ×6, ×8, ×10, ×12, ×14, ×16 |
| Rotation | 5°, 10°, 15°, 20°, 30°, 45°, 60°, 90° |
| Scale | 0.9, 0.8, 0.75, 0.6, 0.5 |
| Crop | 0.9, 0.7, 0.5, 0.3, 0.1 |
| Resize | 0.5, 0.25, 0.1 |

**产出文件：**
```
experiment_output/sweep/
├── sweep_results.json
├── sweep.txt
├── fig2_sweep.png              # 2行×4列子图折线图
```

### 4.5 几何攻击专项 (`geometric`) — 图4

FDW vs GS 在旋转和缩放攻击下的专项对比（比 baseline 更细粒度）。

**扫描范围：**
- 旋转：5°, 10°, 15°, 20°, 30°, 45°, 60°, 90°
- 缩放：0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50

**产出文件：**
```
experiment_output/geometric/
├── geometric_results.json
├── geometric.txt
├── fig4_geometric.png          # 双子图：旋转对比 + 缩放对比
```

### 4.6 FD-Init λ 扫描 (`lambda_sweep`) — §7.2

**扫描范围：** λ ∈ {0.00, 0.04, 0.08, 0.12, 0.16, 0.20}

**测试攻击：** clean, jpeg_25, gauss_blur_4, gauss_noise_005, crop_060, rotate_15, rotate_45, scale_075

**产出文件：**
```
experiment_output/lambda_sweep/
├── lambda_sweep_results.json
├── lambda_sweep.txt
├── fig_lambda_sweep.png        # 每攻击一条线的折线图
```

### 4.7 模板强度 γ 扫描 (`gamma_sweep`) — §7.3

**扫描范围：** γ ∈ {0, 2, 4, 8, 12, 16}

**测试攻击：** clean, rotate_15, rotate_45, scale_075
**额外指标：** 每个γ值采样10张图计算 CLIP-Score

**产出文件：**
```
experiment_output/gamma_sweep/
├── gamma_sweep_results.json
├── gamma_sweep.txt
├── fig_gamma_sweep.png
```

### 4.8 模板注入时间步 t* 扫描 (`tstar_sweep`) — §7.4

**扫描范围：** t* ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}

**测试攻击：** clean, rotate_15, rotate_45, scale_075
**额外指标：** 每个t*值采样10张图计算 CLIP-Score

**产出文件：**
```
experiment_output/tstar_sweep/
├── tstar_sweep_results.json
├── tstar_sweep.txt
├── fig_tstar_sweep.png
```

### 4.9 图像质量 (`quality`) — 表5

10 组 CLIP Score 对比（ViT-L-14），附 t-test 检验。

**对比方法：** FDW / GS / DwtDct / Clean (无水印)

**产出文件：**
```
experiment_output/quality/
├── quality_results.json
├── quality.txt
```

### 4.10 攻击示例图 (`fig1`) — 图1

2×5 网格展示10种攻击效果（无需 --N 参数）。

**产出文件：** `experiment_output/figures/fig1_attack_examples.png`

### 4.11 视觉质量图 (`fig5`) — 图5

4方法 × N prompt 的生成图 + ×10 残差图（无需 --N 参数）。

**产出文件：** `experiment_output/figures/fig5_visual_quality.png`

### 4.12 载荷效果图 (`fig6`) — 图6

3种载荷 (256b / 512b / 1024b / 2048b) × N prompt 的视觉效果（无需 --N 参数）。

**产出文件：** `experiment_output/figures/fig6_payload_visual.png`

---

## 五、完整输出目录结构

```
experiment_output/
├── baseline/
│   ├── baseline_results.json
│   ├── table2_per_attack.txt
│   ├── baseline.txt
│   └── fig3_baseline.png
├── capacity_sweep/
│   ├── capacity_sweep_results.json
│   └── capacity_sweep.txt
├── ablation/
│   ├── ablation_results.json
│   └── ablation.txt
├── sweep/
│   ├── sweep_results.json
│   ├── sweep.txt
│   └── fig2_sweep.png
├── geometric/
│   ├── geometric_results.json
│   ├── geometric.txt
│   └── fig4_geometric.png
├── lambda_sweep/
│   ├── lambda_sweep_results.json
│   ├── lambda_sweep.txt
│   └── fig_lambda_sweep.png
├── gamma_sweep/
│   ├── gamma_sweep_results.json
│   ├── gamma_sweep.txt
│   └── fig_gamma_sweep.png
├── tstar_sweep/
│   ├── tstar_sweep_results.json
│   ├── tstar_sweep.txt
│   └── fig_tstar_sweep.png
├── quality/
│   ├── quality_results.json
│   └── quality.txt
└── figures/
    ├── fig1_attack_examples.png
    ├── fig5_visual_quality.png
    └── fig6_payload_visual.png
```

---

## 六、规模建议

| 用途 | N | 耗时 |
|------|---|------|
| 快速验证 | 2-10 | 5-30 分钟 |
| 开发调试 | 50-100 | 1-3 小时 |
| 正式实验 | 1000 | 2-3 天/实验 |

---

## 七、论文产出对应表

| 论文内容 | 实验 | 输出文件 |
|----------|------|---------|
| 表1：基线对比总表 | baseline | `table1_summary.txt`（需先运行quality） |
| 表2：逐攻击详细对比 | baseline | `table2_per_attack.txt` |
| 表3：容量-冗余权衡 | capacity_sweep | `capacity_sweep.txt` |
| 表4：组件消融 | ablation | `ablation.txt` |
| 表5：图像质量 t-test | quality | `quality.txt` |
| 图1：攻击示例 | fig1 | `fig1_attack_examples.png` |
| 图2：攻击强度扫描 | sweep | `fig2_sweep.png` |
| 图3：基线折线图 | baseline | `fig3_baseline.png` |
| 图4：几何攻击对比 | geometric | `fig4_geometric.png` |
| 图5：视觉质量+残差 | fig5 | `fig5_visual_quality.png` |
| 图6：载荷视觉效果 | fig6 | `fig6_payload_visual.png` |
| §7.2 λ 扫描表+图 | lambda_sweep | `lambda_sweep.txt` + `fig_lambda_sweep.png` |
| §7.3 γ 扫描表+图 | gamma_sweep | `gamma_sweep.txt` + `fig_gamma_sweep.png` |
| §7.4 t* 扫描表+图 | tstar_sweep | `tstar_sweep.txt` + `fig_tstar_sweep.png` |
