# comfyui-moss-accel

**[English](README.md)** | **简体中文** · **[Hugging Face（工作流与对比图）](https://huggingface.co/kefallrain/comfyui-moss-accel)**

面向 MOSS-Image 的零依赖 ComfyUI 节点（**MOSS-Image Accelerate**），通过减少每张图的 DiT 前向次数提速。当 `CFG > 1` 时，常规采样每步需要正、负向两次前向；本节点在完成前段全量 CFG 热身后，让后续排程步只算正向（`C` 步复用引导差，局部重绘 `N` 步省略引导），或在去噪 `x0` 空间按 `sigma` 线性外推整步跳过（`S` 步）。

在随附的 MOSS-Image 双段工作流（`1024 × 1024` 25 步底图 → Lanczos 放大至 `1456 × 1456`、去噪 `0.35` 跑 7 步放大段，RTX 5080，14 格前后不加速夹测）中，`fast` + `refine` 将单张中位耗时从 **`16.20 s` 压到 `10.58 s`（吞吐提速 +53.1%）**，亮度拉普拉斯锐度比 **`0.985×`**，2× 放大下水粉厚涂与 90s 赛璐璐风格 LoRA 笔触无冲淡。同种子成图与不加速结果接近（`RGB rel-L2 0.0645`），但并非逐像素一致。

![双段实测与边界](https://huggingface.co/kefallrain/comfyui-moss-accel/resolve/main/assets/02-benchmark-and-limits-zh.png)

## 安装

将本目录复制或软链接到 `ComfyUI/custom_nodes/` 后重启 ComfyUI。除 ComfyUI 本体外无额外依赖（基于 v0.46.0 V3 节点接口编写）。

## 用法

接在采样器前的 `MODEL` 连线上，位于模型加载器、LoRA 和 `ModelSamplingAuraFlow` 之后：

```
UNETLoader -> ModelSamplingAuraFlow -> MOSS-Image Accelerate -> KSampler
```

本插件同时附带两个零依赖的 Qwen3.5-9B 瘦身文本编码器加载节点（剔除 T2I 阶段无用的 `lm_head` / `visual` / `mtp` 死权重）：
- **`Load MOSS Row-W8A16 Slim Text Encoder`**（`MOSSRowW8A16CLIPLoader`）：加载 `7.94 GB` 每行 INT8 权重 + BF16 激活版本；在 ComfyUI `v0.46+` 下原生接入 `aimdo` `ModelMMAP` 与 `ModelPatcherDynamic`（冷组装仅 `0.73 s`、CPU 私有内存增量 `+0.13 GB`，老版本自动回退流式零 `mmap` 读取），并绕开 `layer.load_state_dict(..., strict=False)` 在 `v0.46` 下触发的 22 GB FP32 补零内存泄漏（见 [Comfy-Org/ComfyUI#18019](https://github.com/Comfy-Org/ComfyUI/issues/18019)）。
- **`Load MOSS Slim Text Encoder (W4A8 / FP8)`**（`MOSSSlimCLIPLoader`）：`<1 ms` 仅读文件头识别键位，直接加载 `4.93 GB` 瘦身 W4A8/FP8 版本（适合 8–12 GB 显卡）。

每次采样输出一行日志，例如：
`MOSS-Image Accelerate: preset fast, 25 steps (coverage 1.00, 25.00 equivalent steps), plan FFFFFFFFFCSFCSFCSFCSFCSFF, forwards 35 of 50 (plan 35), served F15 C5 S5`。
其中 `coverage` 表示本次采样覆盖完整噪声曲线的比例，见下文「局部去噪」。

## 示例工作流

![双段工作流](https://huggingface.co/kefallrain/comfyui-moss-accel/resolve/main/assets/01-workflow-architecture-zh.png)

`workflows/moss_accel_two_pass.json`（UI 格式）与 `workflows/moss_accel_two_pass.api.json`（API 格式）对应同一张图：`1024 × 1024` 25 步底图后经 Lanczos 放大到 `1456 × 1456`，在 `denoise 0.35` 下跑 7 步放大段。底图采样器前接 `fast`，放大采样器前接专为局部去噪设计的 `refine`。两段末尾均带预览或保存；设为 `off` 即恢复该段原生采样。

UI 工作流另带一组默认旁路的同种子 A/B 对照模块（跑 `25 + 9` 原生采样并接入对比节点），日常出图无额外开销，按需开启即可逐段核对。模型文件名均为占位符（`moss_image.safetensors`、`qwen3.5_9b.safetensors`、`flux2_vae.safetensors`、`style_lora.safetensors`），加载后请替换成本机实际文件名。

`workflows/moss_base50_fast50_two_pass.json`（UI）与 `workflows/moss_base50_fast50_two_pass.api.json`（API）演示 **MOSS-Image-Base 50 步 200 万像素消费级套件**：`MOSSRowW8A16CLIPLoader`（`7.94 GB` 瘦身编码器）+ `MOSS-Image-Base` INT8 + `SkipLayerGuidanceDiTSimple`（block 22）+ 一段 `fast_50`（`1248 × 832`，50 步 `euler / simple`，CFG 3.0，`48/100` 次前向）→ `1.4×` Lanczos（`1728 × 1152`）→ 二段 `refine`（9 步 `denoise 0.35`，`12/18` 次前向）。RTX 4060 Ti 16 GB（32 GB 内存）实测两遍 200 万像素均速 **`36.3 s`**（叠加 Base 原生风格 LoRA 为 `37.2–39.0 s`）——比不加速单跑一百万像素原图（`45.5 s`）**反快 20.3%**，像素翻倍。

`workflows/moss_opd12_refine7_poster.json`（UI）与 `workflows/moss_opd12_refine7_poster.api.json`（API）演示快速海报双段组合：一段跑 `832 × 1216` 12 步 `CFG 1.0`（`euler / normal`）OPD 蒸馏加速 LoRA（`moss_opd_fast12_v1.safetensors`），经 Lanczos 放大到 `1152 × 1680` 后，二段在未蒸馏底模上接 `refine` 跑 7 步 `denoise 0.35`（`CFG 2.5`，`er_sde / simple`，RTX 5080 端到端 `6.10 s`）。

## 预设档位

`F` 为完整 CFG 步（2 次前向），`C` 为只算正向并复用上一次引导差（1 次前向），`S` 为跳过模型并在 `x0` 空间按 `sigma` 线性外推（0 次前向），`N` 为只算正向且不加引导（1 次前向，仅 `refine` 与 `custom` 使用）：

| 预设 | 热身比例 | 循环排程 | 8 步 | 15 步 | 20 步 | 25 步 | 30 步 | 50 步 |
|---|---|---|---|---|---|---|---|---|
| `light` | 0.32 | `CF` | 16/16, 0% | 25/30, 17% | 34/40, 15% | 42/50, 16% | 50/60, 17% | 83/100, 17% |
| `balanced`（默认） | 0.32 | `CCF` | 16/16, 0% | 24/30, 20% | 32/40, 20% | 39/50, 22% | 47/60, 22% | 78/100, 22% |
| `fast` | 0.36 | `CSF` | 16/16, 0% | 22/30, 27% | 29/40, 28% | 35/50, 30% | 42/60, 30% | 69/100, 31% |
| `fast_50` | 0.24 | 分段（`SSCSSF` → `SCSCSCSF` → `CNNNNF`） | 回退 `fast` | 回退 `fast` | 回退 `fast` | 回退 `fast` | 31/60, 48% | **48/100, 52%** |
| `refine` | 0.2 | `N` | 仅用于局部去噪段 | | | | | |

相同排程下，若 `S` 步只做零阶保持（`hold`），底图锐度会跌至 `0.951`；改为 `x0` 空间按 `sigma` 线性外推（`linear`）后，锐度回到 `1.006`，耗时与漂移不变。

### `refine`（二段局部重绘专用）

`refine` 用于底图已定好构图的低去噪放大段：前 20% 步数跑完整 CFG，中间全部跑只算正向的 `N` 步，最后一步跑完整 CFG。在每步注入随机噪声的 `er_sde` 下，相邻步引导差相关性很低（`r ≈ 0.15`），局部重绘中段直接省去负向引导比复用旧引导漂移更小。**切勿用于第一段满去噪**：满去噪中段仍靠引导决定人物与构图，省去引导会导致换人（见下图第 `12` 格）。

## 实测数据

RTX 5080 + ComfyUI 0.46.0，MOSS-Image `int8` ConvRot，`er_sde` / `simple`，CFG 2.5，shift 5，14 格提示词前后夹测（`A` 与 `A2` 重复计时噪声 `0.1%`）：

- **单独 9 步放大段（`denoise 0.35`）**：`refine` 提速 **46.2%**，图像 RGB 相对 L2 漂移 **`0.0094`**（仅为 EasyCache 0.2 `0.0176` 的一半），锐度比 **`0.999`**（EasyCache 为 `0.974`）。
- **双段端到端（对比 `25 + 9` 不加速基线 `16.20 s`）**：

| 组合 | 步数 | 端到端中位耗时 | 吞吐提速 | 图像漂移 (RGB L2) | 锐度比 |
|---|---|---|---|---|---|
| `fast` + `refine` | 25 + 9 | 11.36 s | 42.7 % | 0.0615 | 1.004 |
| `fast` + `refine`（示例工作流默认） | 25 + 7 | 10.58 s | 53.1 % | 0.0645 | 0.985 |
| `fast` + `refine` | 25 + 6 | 10.18 s | 59.2 % | 0.0675 | 0.970 |
| Spectrum M1 + `refine` | 25 + 9 | 10.64 s | 52.5 % | 0.0858 | 0.952 |

![同种子局部放大](https://huggingface.co/kefallrain/comfyui-moss-accel/resolve/main/assets/03-visual-crops-and-pitfalls-zh.png)

### 风格 LoRA 保真（2× 放大）

在水粉厚涂与 90 年代手绘赛璐璐两款风格 LoRA 上各测 6 格：`fast` + `refine`（`11.4 s`，+42.7%）锐度比保持 **`1.000`** 与 **`0.988`**，刮刀笔触与排线颗粒保留完整；叠加社区 `8 + 3` 步 DMD 蒸馏 LoRA（`3.3 s`）时，锐度比跌至 **`0.473`** 与 **`0.452`**，画风肌理明显被抹平。

![风格 LoRA 局部放大](https://huggingface.co/kefallrain/comfyui-moss-accel/resolve/main/assets/04-style-lora-2x-zoom-zh.png)

### MOSS-Image-Base 50 步（`fast_50`）与 200 万像素消费级套件实测

在 **RTX 4060 Ti 16 GB（32 GB 内存，ComfyUI `v0.46.0`）** 上基于 50 步 **MOSS-Image-Base**（`moss_image_base_int8_convrot.safetensors`，`euler / simple`，CFG 3.0，shift 5.0，`SkipLayerGuidanceDiTSimple` block 22）实测：

| 流程（RTX 4060 Ti 16 GB，MOSS-Image-Base INT8 + Row-W8A16 瘦身 TE） | 分辨率 | DiT 前向次数 | 耗时 | 对比 1MP 原版 |
|---|---|---:|---:|---|
| 原生 50 步（`preset=off`，单段暖机） | `1248 × 832`（1.04 MP） | 100 | 44.4–45.5 s | 基线 |
| `fast_50` 50 步分段（`FFFFFFFFFFFF` + `SSCSSF`×4 + `SCSCSCSF` + `CNNNNF`） | `1024 × 1024`（1.05 MP） | **48** | **21.67 s** | **2.05× 提速**（`rel-L2 0.114`，锐度 `1.024×`） |
| 两遍 `fast_50` + `1.4× Lanczos` + `refine`（9 步 `denoise 0.35`，无 LoRA） | `1728 × 1152`（2.0 MP） | 48 + 12 | **36.3–36.8 s** | **像素翻倍仍快 20.3%** |
| 两遍 `fast_50` + `refine` + Base 原生风格 LoRA（`LoraLoaderModelOnly`） | `1728 × 1152` / `1152 × 1728`（2.0 MP） | 48 + 12 | **37.2–39.0 s** | **像素翻倍仍快 14–18%** |

- **32 GB 内存 Windows 实测**：`MOSSRowW8A16CLIPLoader` 接入 `v0.46+` 原生 `aimdo` `ModelMMAP` 并采用点对点叶参数赋值后，32 层编码器冷组装仅需 **`0.73 s`**、CPU 私有提交增量仅 **`+0.13 GB`**（全流程峰值 `16.8 GB`，32 GB 内存整机仍余 `21.3 GB` 虚拟内存；绕开 [Comfy-Org/ComfyUI#18019](https://github.com/Comfy-Org/ComfyUI/issues/18019)）。
- **风格 LoRA 搭配提示**：在 **MOSS-Image-Base**（50 步 + SLG 22）上优先搭配 Base 原生训练的风格 LoRA（用 `LoraLoaderModelOnly`，切勿用 Bypass 加载器）；纯 SFT 训练的 LoRA 在 Base 上可能出现肢体漂移。

## 局部去噪与等效步数密度

局部去噪（img2img / hires-fix）不能只按面板步数判断是否可加速：`denoise 0.35` 下跑 7 步，实际截取的是 20 步完整网格（`int(7 / 0.35)`）的最后 7 步，步间距与 20 步完整采样同样细密。节点根据首个 `sigma` 在模型自身的 `percent_to_sigma` 曲线上二分反查覆盖比例 `coverage`，当等效步数 `steps / coverage >= 11.75` 且实际步数 `>= 6` 时启用排程；而从头跑的 8 步蒸馏流（`coverage = 1.0`，等效 `8` 步）会自动保持全算。

## 自动旁路条件（何时不介入）

- 等效步数少于 `11.75` 或实际步数少于 `6`（如 8 步蒸馏流）。
- 非 Flow-Matching 采样模型。
- 单步多次求值的二阶采样器（`heun`、`dpm_2` 等）：首次检测到重复或偏离 `sigma` 起自动切回全算。
- `CFG = 1` 或无引导采样器（`BasicGuider`）：本身无负向分支可省，自动全算并记录日志。

## 局限与边界

- **非逐像素一致**：同种子下耳环、衣扣、轻微偏头角度等细部会有位移（`RGB rel-L2 ≈ 0.065`）；需严格复现原图基线时请设为 `off` 或开启工作流内置 A/B 对照。
- **后半程负面提示词变弱**：`C` 步复用早期负向引导，`N` 步不跑负向分支，采样后半段依赖负面词修手或压瑕疵的权重会减弱。
- **`refine` 仅限局部去噪段**：在第一段满去噪上使用 `refine`（`N` 步）会导致人物或主体改变。
- **标定范围与异常值**：预设基于 MOSS-Image（`int8` ConvRot）+ `er_sde` 标定，其他模型或采样器未单独调参；节点不做 `NaN` 检查，若某步出现异常值会传播到后续复用步。

## 测试

```sh
./check
COMFYUI_PATH=/path/to/ComfyUI MOSS_ACCEL_PYTHON=/path/to/comfy/venv/bin/python ./check
```
