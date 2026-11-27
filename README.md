# comfyui-moss-accel

**English** | **[简体中文](README.zh-CN.md)** · **[Hugging Face (workflows & figures)](https://huggingface.co/kefallrain/comfyui-moss-accel)**

A zero-dependency ComfyUI node, **MOSS-Image Accelerate**, that runs fewer DiT forwards per image on MOSS-Image. With CFG above 1 every sampling step normally costs two forwards, one for the positive prompt and one for the negative. After a full-CFG warmup, the node lets scheduled steps evaluate the positive prompt only (`C` reusing guidance, or `N` unguided on a partial-denoise pass), or skip the model and extrapolate linearly over sigma in denoised `x0` space (`S`).

On the included two-pass MOSS-Image workflow (`1024 × 1024` 25-step base → Lanczos `1456 × 1456` 7-step refine at `denoise 0.35`, RTX 5080, 14 prompts with interleaved unaccelerated controls), `fast` + `refine` cuts median generation time from **`16.20 s` to `10.58 s` (+53.1% throughput)** at **`0.985×`** sharpness, keeping painterly and cel style LoRAs intact at 2× zoom. The image stays close to the unaccelerated run at the same seed (`RGB rel-L2 0.0645`), though it is not pixel-identical.

![Benchmark and limits](https://huggingface.co/kefallrain/comfyui-moss-accel/resolve/main/assets/02-benchmark-and-limits-en.png)

## Install

Copy or link this directory into `ComfyUI/custom_nodes/` and restart ComfyUI. It needs nothing beyond ComfyUI itself (written against v0.46.0, V3 node API).

## Use

Put the node on the MODEL line just before the sampler, after the model loader, LoRAs and `ModelSamplingAuraFlow`:

```
UNETLoader -> ModelSamplingAuraFlow -> MOSS-Image Accelerate -> KSampler
```

The package also includes two zero-dependency slim Qwen3.5-9B text encoder loaders (for checkpoints stripped of unused `lm_head`, `visual`, and `mtp` weights):
- **`Load MOSS Row-W8A16 Slim Text Encoder`** (`MOSSRowW8A16CLIPLoader`): loads the `7.94 GB` per-row INT8 + BF16-activation checkpoint via ComfyUI `v0.46+` file-backed `aimdo` `ModelMMAP` + `ModelPatcherDynamic` (`0.73 s` assembly, `+0.13 GB` CPU private commit; automatically falls back to buffered zero-mmap streaming on older ComfyUI). Directly assigns non-quantized leaf parameters instead of calling `layer.load_state_dict(..., strict=False)` to avoid ComfyUI `v0.46`'s ~22 GB FP32 zero-fill RAM spike ([Comfy-Org/ComfyUI#18019](https://github.com/Comfy-Org/ComfyUI/issues/18019)).
- **`Load MOSS Slim Text Encoder (W4A8 / FP8)`** (`MOSSSlimCLIPLoader`): header-only `<1 ms` safetensors key inspection for `4.93 GB` stripped W4A8/FP8 checkpoints on 8–12 GB GPUs.

Each sampling logs one line, for example
`MOSS-Image Accelerate: preset fast, 25 steps (coverage 1.00, 25.00 equivalent steps), plan FFFFFFFFFCSFCSFCSFCSFCSFF, forwards 35 of 50 (plan 35), served F15 C5 S5`.
`coverage` is the share of the model's whole schedule the run covers; see [Partial denoise](#partial-denoise).

## Example workflows

![Two-pass workflow](https://huggingface.co/kefallrain/comfyui-moss-accel/resolve/main/assets/01-workflow-architecture-en.png)

`workflows/moss_accel_two_pass.json` (ComfyUI UI format) and `workflows/moss_accel_two_pass.api.json` (API format) are the same graph: a two-pass MOSS-Image run, a `1024 × 1024` base image at 25 steps followed by a lanczos upscale to `1456 × 1456` and a 7-step pass at `denoise 0.35`. It carries two acceleration nodes: the base sampler runs behind one set to `fast`, the upscale sampler behind a second one set to `refine`, the preset made for a partial-denoise pass. Each pass ends in a preview or save, so the base image can be judged before the upscale spends its steps. Change `preset` on either node to try the others; set a node to `off` to get the plain sampler for that pass.

The UI file also holds an A/B module that runs the same seeds through the plain `25 + 9` sampler and puts each stage next to the accelerated one in an image compare node. Those nodes are bypassed, so a normal run costs nothing extra; enable them to compare. The model file names are placeholders (`moss_image.safetensors`, `qwen3.5_9b.safetensors`, `flux2_vae.safetensors`, `style_lora.safetensors`): pick your own files in the loader nodes. The style LoRA node is bypassed by default.

`workflows/moss_base50_fast50_two_pass.json` (UI) and `workflows/moss_base50_fast50_two_pass.api.json` (API) demonstrate the **MOSS-Image-Base 50-step 2-Megapixel consumer stack**: `MOSSRowW8A16CLIPLoader` (`7.94 GB` slim TE) + `MOSS-Image-Base` INT8 + `SkipLayerGuidanceDiTSimple` (block 22) + Pass 1 `fast_50` (`1248 × 832`, 50 steps `euler / simple`, CFG 3.0, `48/100` forwards) → `1.4×` Lanczos (`1728 × 1152`) → Pass 2 `refine` (9 steps `denoise 0.35`, `12/18` forwards). On an RTX 4060 Ti 16 GB (32 GB RAM), full 2-pass 2 MP renders average **`36.3 s`** (`37.2–39.0 s` with a Base-native style LoRA) — **20.3% faster than a single unaccelerated 1 MP render (`45.5 s`)**, at double the pixel count.

`workflows/moss_opd12_refine7_poster.json` (UI) and `workflows/moss_opd12_refine7_poster.api.json` (API) pair a 12-step `CFG 1.0` (`euler / normal`) on-policy distilled Pass 1 (`832 × 1216`, `moss_opd_fast12_v1.safetensors`) with a Lanczos upscale to `1152 × 1680` and a 7-step un-distilled `refine` Pass 2 (`CFG 2.5`, `er_sde / simple`, `denoise 0.35`, `6.10 s` end-to-end on an RTX 5080).

## Presets

`F` is a full-CFG step (two forwards), `C` a positive-only step that reuses the guidance (one forward), `S` a skipped step (none). `N` is a positive-only step with no guidance (one forward, [Unguided steps](#unguided-steps)); `refine` uses it, and `custom` takes it. Forwards per image against the plain sampler, with the share saved:

| Preset | Warmup | Pattern | 8 steps | 15 | 20 | 25 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|
| `light` | 0.32 | `CF` | 16/16, 0% | 25/30, 17% | 34/40, 15% | 42/50, 16% | 50/60, 17% | 83/100, 17% |
| `balanced` (default) | 0.32 | `CCF` | 16/16, 0% | 24/30, 20% | 32/40, 20% | 39/50, 22% | 47/60, 22% | 78/100, 22% |
| `fast` | 0.36 | `CSF` | 16/16, 0% | 22/30, 27% | 29/40, 28% | 35/50, 30% | 42/60, 30% | 69/100, 31% |
| `fast_50` | 0.24 | phased (`SSCSSF` → `SCSCSCSF` → `CNNNNF`) | — (`fast`) | — (`fast`) | — (`fast`) | — (`fast`) | 31/60, 48% | **48/100, 52%** |
| `refine` | 0.2 | `N` | [partial denoise only](#refine) | | | | | |

Sampling time follows the forward count closely, since a forward is almost all of a step. [Measured](#measured) has the times and the image checks.

`light` and `balanced` have no `S` step. `fast` skips by `linear`: an `S` step extrapolates the last two evaluated predictions over sigma in `x0` space instead of repeating the last one. On the same plan, `hold` left the images at `0.951` of the unaccelerated sharpness and `linear` at `1.006`, with the same speed-up and the same drift from the unaccelerated image.

`off` leaves the model as it is, and also removes an accelerator applied earlier on the same line.

### Refine

`refine` is for the second, low-denoise pass of an upscale or hires fix, where the base image already holds the composition. After a warmup of a fifth of the steps every step runs the positive prompt only, with no guidance, except the last. Do not use it on a full-denoise first pass: there the guidance still decides the content, and leaving it out changes what the image shows ([Unguided steps](#unguided-steps)). Like every preset it plans by the density of the schedule the run was cut from ([Partial denoise](#partial-denoise)), so a pass of fewer than 6 steps, or one cut from fewer than 12, runs in full. Forwards per pass against the plain sampler:

| Run | Steps | Coverage | Equivalent | Plan with `refine` | Forwards |
|---|---|---|---|---|---|
| `denoise` 0.6 | 15 | 0.6 | 25 | `FFFNNNNNNNNNNNF` | 19/30, 37% |
| `denoise` 0.35 | 9 | 0.36 | 25 | `FFNNNNNNF` | 12/18, 33% |
| `denoise` 0.35 | 7 | 0.35 | 20 | `FFNNNNF` | 10/14, 29% |
| `denoise` 0.25 | 6 | 0.25 | 24 | `FFNNNF` | 9/12, 25% |
| `denoise` 0.2 | 5 | 0.2 | 25 | `FFFFF` | 10/10, 0% |

The plans are `plan_steps` output at those steps and coverages. `denoise` 0.25 at 6 steps is the six-step tail of a 24-step grid; `denoise` 0.35 at 7 steps (`FFNNNNF`) is the tail of the example workflow.

## Measured

Same-seed runs against the plain sampler on 14 prompt and seed cells (twelve at `1024 × 1024`, two at `832 × 1248`): the base pass at 25 steps, CFG 2.5, `er_sde` / `simple`, shift 5, and a 9-step `denoise 0.35` upscale tail (`1456 × 1456`). EasyCache is ComfyUI's own node, for reference.

| | Speed-up, base / tail | Drift (latent), base / tail | Sharpness, base / tail |
|---|---|---|---|
| `light` | 18.7 % / 18.5 % | 0.135 / 0.044 | 1.003 / 1.002 |
| `balanced` | 28.1 % / 25.9 % | 0.160 / 0.049 | 1.001 / 1.001 |
| `fast` | 42.5 % / 26.6 % | 0.154 / 0.047 | 1.006 / 1.002 |
| EasyCache 0.2 | 29.7 % / 45.2 % | 0.116 / 0.067 | 0.954 / 0.980 |

Speed-up is the plain time over the accelerated time, minus 1, as the median over the cells. Drift here is the relative L2 distance of the final latent from the plain run of the same seed; the plain run repeated gives 0. Sharpness is the mean absolute Laplacian of the decoded image's luma as a ratio to the plain image's: EasyCache moves details less on average and softens them (and occasionally smears portrait faces on the base pass, see cell `11` below), whereas the presets keep sharpness near `1.00`.

### Second pass with `refine` and end-to-end results

Measured on an RTX 5080 with ComfyUI 0.46.0 across the same 14 cells, with the plain run (`A`) measured before and after (`A2`, repeat timing noise `0.1%`). Here drift is the relative L2 distance of the decoded RGB image from the plain run.

On the 9-step `denoise 0.35` tail alone, `refine` (`warmup 0.2`, pattern `N`) gives **46.2% speed-up**, RGB drift **`0.0094`** (half of EasyCache 0.2's `0.0176`), and sharpness **`0.999`** (vs EasyCache `0.974`). Over both passes against the plain `25 + 9` baseline (`16.20 s` median):

| Base pass, then `refine` on the tail | Steps | End-to-end | Speed-up | Drift (RGB) | Sharpness |
|---|---|---|---|---|---|
| `fast` + `refine` | 25 + 9 | 11.36 s | 42.7 % | 0.0615 | 1.004 |
| `fast` + `refine` (example workflow) | 25 + 7 | 10.58 s | 53.1 % | 0.0645 | 0.985 |
| `fast` + `refine` | 25 + 6 | 10.18 s | 59.2 % | 0.0675 | 0.970 |
| Spectrum M1 + `refine` | 25 + 9 | 10.64 s | 52.5 % | 0.0858 | 0.952 |

![Same-seed crops](https://huggingface.co/kefallrain/comfyui-moss-accel/resolve/main/assets/03-visual-crops-and-pitfalls-en.png)

### Style LoRA fidelity

Tested across 6 prompts each on two contrasting style LoRAs (thick painterly gouache and 1990s hand-inked cel animation). `fast` + `refine` (`11.4 s`, +42.7%) kept sharpness at **`1.000`** and **`0.988`** with brushwork and cel grain intact at 2× zoom; an `8 + 3` community DMD LoRA run (`3.3 s`) dropped sharpness to **`0.473`** and **`0.452`**, smoothing out the style texture.

![Style LoRA at 2x zoom](https://huggingface.co/kefallrain/comfyui-moss-accel/resolve/main/assets/04-style-lora-2x-zoom-en.png)

### MOSS-Image-Base 50-step (`fast_50`) & 2-Megapixel Consumer Suite

Calibrated directly on 50-step **MOSS-Image-Base** (`moss_image_base_int8_convrot.safetensors`, `euler / simple`, CFG 3.0, shift 5.0, `SkipLayerGuidanceDiTSimple` block 22) across interleaved A-B-A sweeps on an **RTX 4060 Ti 16 GB (32 GB RAM, ComfyUI `v0.46.0`)**:

| Pipeline (RTX 4060 Ti 16 GB, MOSS-Image-Base INT8 + Row-W8A16 Slim TE) | Resolution | Forwards | Time | vs 1 MP Plain |
|---|---|---:|---:|---|
| Plain 50-step (`preset=off`, 1-pass warm) | `1248 × 832` (1.04 MP) | 100 | 44.4–45.5 s | baseline |
| `fast_50` 50-step (`FFFFFFFFFFFF` + `SSCSSF`×4 + `SCSCSCSF` + `CNNNNF`) | `1024 × 1024` (1.05 MP) | **48** | **21.67 s** | **2.05× faster** (`rel-L2 0.114`, sharp `1.024×`) |
| Two-pass `fast_50` + `1.4× Lanczos` + `refine` (9 steps, `denoise 0.35`, No-LoRA) | `1728 × 1152` (2.0 MP) | 48 + 12 | **36.3–36.8 s** | **20.3% faster at 2× pixels** |
| Two-pass `fast_50` + `refine` + Base-native Style LoRA (`LoraLoaderModelOnly`) | `1728 × 1152` / `1152 × 1728` (2.0 MP) | 48 + 12 | **37.2–39.0 s** | **14–18% faster at 2× pixels** |

- **Slim TE memory on 32 GB RAM Windows (`ComfyUI v0.46+`)**: Using `MOSSRowW8A16CLIPLoader` with native `aimdo` `ModelMMAP` + direct leaf assignment keeps 32-layer TE assembly at **`0.73 s`** and **`+0.13 GB`** CPU private commit (peak process commit `16.8 GB`, leaving `21.3 GB` free virtual memory on a 32 GB RAM system, vs `38.7 GB` OOM when calling `layer.load_state_dict(..., strict=False)` — [Comfy-Org/ComfyUI#18019](https://github.com/Comfy-Org/ComfyUI/issues/18019)).
- **Style LoRA compatibility note**: When running **MOSS-Image-Base** (`50` steps, `CFG 3.0`, `SLG 22`), pair with LoRAs trained natively on `MOSS-Image-Base` (via `LoraLoaderModelOnly`, never `LoraLoaderBypassModelOnly`); LoRAs trained strictly on `MOSS-Image-SFT` can show anatomy drift on Base.

## Custom

With `preset` set to `custom`, four optional inputs take effect; with any other preset they are ignored.

| Input | Values | Meaning |
|---|---|---|
| `warmup` | 0.2 to 0.9 | Share of the steps, from the start, that always run full CFG. At least two steps. |
| `pattern` | up to 16 of `F`, `C`, `N`, `S` | Repeated after the warmup. No two `S` in a row, also across the end of the pattern. `N` and `S` do not mix. |
| `skip_mode` | `hold`, `linear` | An `S` step repeats the last prediction, or extrapolates the last two over sigma. |
| `cache_space` | `x0`, `v` | Hold and reuse guidance on the denoised prediction, or on the flow velocity. |

Every plan keeps the last step full and never skips the step before it. Invalid input is refused when the node runs, with the reason in the error.

`x0` is the default. Under a deterministic first-order sampler such as `euler`, holding `x0` and holding `v` give the same trajectory. Under `er_sde`, which adds noise at every step, a held velocity carries that noise into the prediction and a held `x0` does not.

### Unguided steps

`N` is a positive-only step with no guidance at all: the negative branch is taken equal to the positive one, so the step's CFG output is the positive prediction. It costs one forward, like `C`, and touches neither the guidance a `C` step reuses nor the predictions an `S` step extrapolates from. For that reason `N` and `S` cannot share a pattern: an `S` step would extrapolate from negative branches that were never evaluated.

Under a sampler that injects fresh noise at every step (`er_sde`), the guidance of neighbouring steps is only weakly related (`r ≈ 0.15`), and carrying the last one forward on a partial-denoise pass adds more error than leaving guidance out. On a full-denoise pass, however, guidance still decides subject identity: with warmup `0.32`, patterns `NF`, `NNF` and `NNNF` swapped the person in the image (cell `12` above). Keep `N` for partial-denoise passes only.

## Partial denoise

A partial `denoise` (img2img, a hires-fix pass) is planned on the steps that actually run, and whether it is planned at all is judged by how dense those steps are, not by how many there are.

KSampler at `denoise` 0.35 and 7 steps cuts the last 7 steps of a 20-step grid (`int(7 / 0.35)`), so each step is as close to the next as in a full 20-step run. The node finds `coverage` by binary search on the model's `percent_to_sigma` curve from the run's first sigma, and plans the run when `steps / coverage` reaches `11.75` (12 less scheduler cut tolerance) and `steps >= 6`. A full 8-step run has `coverage 1.0` (`8` equivalent steps) and stays untouched.

## When it does nothing

- Fewer than `11.75` equivalent steps or fewer than `6` steps: every step runs in full, so an 8-step distilled run is untouched.
- A model whose sampling is not flow matching: the node logs that and steps aside.
- A sampler that evaluates the model more than once per step (`heun`, `dpm_2`, and similar): from the first repeated sigma on, every evaluation runs in full.
- CFG 1, or a guider with no CFG (`BasicGuider`): ComfyUI already leaves the negative prompt out, so every step runs in full and the log says why.
- A node applied after this one that replaces `sampler_calc_cond_batch_function` without chaining to it: the log says the accelerator was never called.

## Limits

- **Not pixel-identical**: Same-seed images shift minor details (`RGB rel-L2 ≈ 0.065`: coat clasps, earrings, slight head tilt). Set `preset` to `off` (or enable the workflow's A/B group) when exact baseline reproduction is required.
- **Weaker late negative prompt**: `C` steps give the negative prompt less say, and `N` steps none, so negative prompt terms that matter late in sampling (hands, fine defects) carry less weight after the warmup.
- **Partial denoise only for `refine`**: Using `refine` (`N` steps) on a full-denoise first pass changes who or what appears in the image.
- **Calibration scope & NaNs**: The presets were calibrated on MOSS-Image (`int8` ConvRot) with `er_sde`; other flow-matching models or samplers run the same plan without separate tuning. The node does not check outputs for `NaN`; a bad step spreads to the steps that reuse it.

## Going back

Set `preset` to `off`, or bypass the node. Nothing is stored outside the sampling run.

## Tests

```sh
./check
COMFYUI_PATH=/path/to/ComfyUI MOSS_ACCEL_PYTHON=/path/to/comfy/venv/bin/python ./check
```

The plan tests need only Python. The ComfyUI tests build a small MOSS-Image model with random weights on the CPU and sample with the real ComfyUI samplers; without `COMFYUI_PATH` they are skipped and say why.
