# comfyui-moss-accel

A ComfyUI node, **MOSS-Image Accelerate**, that runs fewer DiT forwards per image on MOSS-Image. With CFG
above 1 every sampling step normally costs two forwards, one for the positive prompt and one for the
negative. After a full-CFG warmup, the node lets scheduled steps evaluate the positive prompt only and reuse
the last guidance, or skip the model and hold the last prediction. The image stays close to the one without
the node at the same seed; it is not identical.

## Install

Copy or link this directory into `ComfyUI/custom_nodes/` and restart ComfyUI. It needs nothing beyond
ComfyUI itself (written against v0.46.0, V3 node API).

## Use

Put the node on the MODEL line just before the sampler, after the model loader, LoRAs and
`ModelSamplingAuraFlow`:

```
UNETLoader -> ModelSamplingAuraFlow -> MOSS-Image Accelerate -> KSampler
```

Each sampling logs one line, for example
`MOSS-Image Accelerate: preset fast, 25 steps (coverage 1.00, 25.00 equivalent steps), plan FFFFFFFFFCSFCSFCSFCSFCSFF, forwards 35 of 50 (plan 35), served F15 C5 S5`.
`coverage` is the share of the model's whole schedule the run covers; see [Partial denoise](#partial-denoise).

## Example workflows

`workflows/moss_accel_two_pass.json` (ComfyUI UI format) and `workflows/moss_accel_two_pass.api.json` (API format)
are the same graph: a two-pass MOSS-Image run, a 1024 x 1024 base image at 25 steps followed by a lanczos
upscale to 1456 x 1456 and a 7-step pass at denoise 0.35. It carries two acceleration nodes: the base sampler
runs behind one set to `fast`, the upscale sampler behind a second one set to `refine`, the preset made for a
partial-denoise pass. Each pass ends in a preview or save, so the base image can be judged before the upscale
spends its steps. Change `preset` on either node to try the others; set a node to `off` to get the plain sampler
for that pass.

The UI file also holds an A/B module that runs the same seeds through the plain 25 + 9 sampler and puts each stage next
to the accelerated one in an image compare node. Those nodes are bypassed, so a normal run costs nothing extra;
enable them to compare. The model file names are placeholders (`moss_image.safetensors`, `qwen3.5_9b.safetensors`,
`flux2_vae.safetensors`, `style_lora.safetensors`): pick your own files in the loader nodes. The style LoRA node is bypassed
by default.

## Presets

`F` is a full-CFG step (two forwards), `C` a positive-only step that reuses the guidance (one forward),
`S` a skipped step (none). `N` is a positive-only step with no guidance (one forward, [Unguided steps](#unguided-steps)); `refine` uses it, and `custom` takes it. Forwards per image against the plain sampler, with the share saved:

| Preset | Warmup | Pattern | 8 steps | 15 | 20 | 25 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|
| `light` | 0.32 | `CF` | 16/16, 0% | 25/30, 17% | 34/40, 15% | 42/50, 16% | 50/60, 17% | 83/100, 17% |
| `balanced` (default) | 0.32 | `CCF` | 16/16, 0% | 24/30, 20% | 32/40, 20% | 39/50, 22% | 47/60, 22% | 78/100, 22% |
| `fast` | 0.36 | `CSF` | 16/16, 0% | 22/30, 27% | 29/40, 28% | 35/50, 30% | 42/60, 30% | 69/100, 31% |
| `refine` | 0.2 | `N` | [partial denoise only](#refine) | | | | | |

Sampling time follows the forward count closely, since a forward is almost all of a step. [Measured](#measured)
has the times and the image checks.

`light` and `balanced` have no `S` step. `fast` skips by `linear`: an `S` step extrapolates the last two
evaluated predictions over sigma instead of repeating the last one. On the same plan, `hold` left the images at
0.951 of the unaccelerated sharpness and `linear` at 1.006, with the same speed-up and the same drift from the
unaccelerated image.

`off` leaves the model as it is, and also removes an accelerator applied earlier on the same line.

### Refine

`refine` is for the second, low-denoise pass of an upscale or hires fix, where the base image already holds the
composition. After a warmup of a fifth of the steps every step runs the positive prompt only, with no guidance, except
the last. Do not use it on a full-denoise first pass: there the guidance still decides the content, and leaving it
out changes what the image shows ([Unguided steps](#unguided-steps)). Like every preset it plans by the density of
the schedule the run was cut from ([Partial denoise](#partial-denoise)), so a pass of fewer than 6 steps, or one cut
from fewer than 12, runs in full. Forwards per pass against the plain sampler:

| Run | Steps | Coverage | Equivalent | Plan with `refine` | Forwards |
|---|---|---|---|---|---|
| `denoise` 0.6 | 15 | 0.6 | 25 | `FFFNNNNNNNNNNNF` | 19/30, 37% |
| `denoise` 0.35 | 9 | 0.36 | 25 | `FFNNNNNNF` | 12/18, 33% |
| `denoise` 0.35 | 7 | 0.35 | 20 | `FFNNNNF` | 10/14, 29% |
| `denoise` 0.25 | 6 | 0.25 | 24 | `FFNNNF` | 9/12, 25% |
| `denoise` 0.2 | 5 | 0.2 | 25 | `FFFFF` | 10/10, 0% |

The plans are `plan_steps` output at those steps and coverages. `denoise` 0.25 at 6 steps is the six-step tail of a 24-step grid; `denoise` 0.35 at 7 steps (`FFNNNNF`) is the tail of the example workflow.

## Measured

Same-seed runs against the plain sampler on 14 prompt and seed cells (twelve at 1024 x 1024, two at 832 x 1248):
the base pass at 25 steps, CFG 2.5, `er_sde` / `simple`, shift 5, and the upscale tail of the example workflow
(lanczos to 1456 x 1456, denoise 0.35, 9 steps). EasyCache is ComfyUI's own node, for reference.

| | Speed-up, base / tail | Drift, base / tail | Sharpness, base / tail |
|---|---|---|---|
| `light` | 18.7 % / 18.5 % | 0.135 / 0.044 | 1.003 / 1.002 |
| `balanced` | 28.1 % / 25.9 % | 0.160 / 0.049 | 1.001 / 1.001 |
| `fast` | 42.5 % / 26.6 % | 0.154 / 0.047 | 1.006 / 1.002 |
| EasyCache 0.2 | 29.7 % / 45.2 % | 0.116 / 0.067 | 0.954 / 0.980 |

Speed-up is the plain time over the accelerated time, minus 1, as the median over the cells. Drift is the relative
L2 distance of the final latent from the plain run of the same seed; the plain run repeated gives 0. Sharpness is the
mean absolute Laplacian of the decoded image's luma as a ratio to the plain image's. Drift says how far details
moved, sharpness whether they went soft: EasyCache moves them less and softens them, the presets move them more and
keep them sharp. Over both passes of the example workflow the presets come to about 17, 24 and 30 % faster; the tail
gains less because nine steps leave room for only a few reused steps after the warm-up.

### Second pass with `refine`

The same kind of run on the tail alone: lanczos from 1024 x 1024 to 1456 x 1456, 9 steps, denoise 0.35, `er_sde`,
CFG 2.5, 14 cells, median of each figure, the plain run (A) measured before and after the accelerated one (A2) so
that a drift of the machine shows; the repeat noise of the speed was 0.1 %. Measured on an RTX 5080 with ComfyUI 0.46.0.
Here drift is the relative L2 distance of the decoded RGB image from the plain run (not the latent distance used in the
table above).

| Tail pass | Speed-up | Drift (RGB) | Sharpness |
|---|---|---|---|
| `custom`, warmup 0.2, pattern `N` (`refine`) | 46.2 % | 0.0094 | 0.999 |

Next to `fast`'s 26.6 % on the same tail, this saves more and moves the details less; the tail is not told
apart from the plain one by eye. The same pattern on the 25-step base pass, with warmup 0.32 and `N`, `NF`, `NNF` or
`NNNF`, replaced the person in the image, which is why `refine` is written for partial denoise only.

Over both passes, the two-pass workflow measured end to end against the plain 25 + 9 baseline (`16.20 s` median):

| Base pass, then `refine` on the tail | End-to-end | Speed-up | Drift (RGB) | Sharpness |
|---|---|---|---|---|
| `fast` + `refine` (9-step tail) | 11.36 s | 42.7 % | 0.0615 | 1.004 |
| `fast` + `refine` (7-step tail, example workflow) | 10.58 s | 53.1 % | 0.0645 | 0.985 |
| `fast` + `refine` (6-step tail) | 10.18 s | 59.2 % | 0.0675 | 0.970 |
| the third-party Spectrum node set by hand + `refine` (9-step tail) | 10.64 s | 52.5 % | 0.0858 | 0.952 |

Trimming the `refine` tail to 7 steps (`FFNNNNF`) beats the third-party Spectrum M1 baseline on speed, drift and
sharpness without extra dependencies, and preserves painterly and cel style LoRAs without dilution at 2x zoom.

## Custom

With `preset` set to `custom`, four optional inputs take effect; with any other preset they are ignored.

| Input | Values | Meaning |
|---|---|---|
| `warmup` | 0.2 to 0.9 | Share of the steps, from the start, that always run full CFG. At least two steps. |
| `pattern` | up to 16 of `F`, `C`, `N`, `S` | Repeated after the warmup. No two `S` in a row, also across the end of the pattern. `N` and `S` do not mix. |
| `skip_mode` | `hold`, `linear` | An `S` step repeats the last prediction, or extrapolates the last two over sigma. |
| `cache_space` | `x0`, `v` | Hold and reuse guidance on the denoised prediction, or on the flow velocity. |

Every plan keeps the last step full and never skips the step before it. Invalid input is refused when the
node runs, with the reason in the error.

`x0` is the default. Under a deterministic first-order sampler such as `euler`, holding `x0` and holding `v`
give the same trajectory. Under `er_sde`, which adds noise at every step, a held velocity carries that noise
into the prediction and a held `x0` does not. Reused guidance differs between the two by the ratio of the
current to the last full step's sigma.

### Unguided steps

`N` is a positive-only step with no guidance at all: the negative branch is taken equal to the positive one, so the
step's CFG output is the positive prediction. It costs one forward, like `C`, and touches neither the guidance a `C`
step reuses nor the predictions an `S` step extrapolates from. For that reason `N` and `S` cannot share a pattern: an
`S` step would extrapolate from negative branches that were never evaluated. Only `refine` uses `N`; try it elsewhere
with `custom`, for instance warmup `0.32` and pattern `NF` or `NNF`.

Where `N` is safe depends on the pass. On a partial-denoise pass the composition is already fixed by the base image,
and `N` steps measured as indistinguishable from the plain run ([Second pass with `refine`](#second-pass-with-refine)).
On a full-denoise pass `N` changes the content: with warmup `0.32`, patterns `NF`, `NNF` and `NNNF` swapped the person
in the image.

Use it where `C` would be reusing a guidance that has gone stale. Under a sampler that injects fresh noise at every
step, such as `er_sde`, the guidance of neighbouring steps is only weakly related, and carrying the last one forward
can add more error than leaving guidance out of that step. Both open-loop trajectory checks and closed-loop 14-cell
renders confirm that `N` steps on a partial-denoise pass cut drift in half against EasyCache 0.2 (`0.0094` vs `0.0176`
RGB relative L2) while keeping full sharpness (`0.999` vs `0.974`). Other samplers are untested. An `N` step gives the
negative prompt no say, so keep the warmup long enough that composition is settled before the first one.

## When it does nothing

- Fewer than 11.75 equivalent steps (12 less a margin: the evenly cut schedulers put the tails of a 12-step grid between
  11.94 and 12, and those of shorter grids at 11.63 and below, except six-step tails that start halfway down, which
  measure like a 12-step grid's), or fewer than 6
  steps: the plan is all full, so an 8-step distilled run is untouched. The schedulers that do not cut a grid evenly
  (karras, exponential, kl_optimal, beta, linear_quadratic) count a tail by its local density, which the log shows
  (15 to 195 equivalent steps measured for karras, exponential and kl_optimal).
- A model whose sampling is not flow matching: the node logs that and steps aside.
- A sampler that evaluates the model more than once per step (`heun`, `dpm_2`, and similar): from the
  first repeated sigma on, every evaluation runs in full.
- CFG 1, or a guider with no CFG (`BasicGuider`): ComfyUI leaves the negative prompt out, so there is nothing
  to save, and a distilled few-step recipe runs at CFG 1 with no steps to spare. Every step runs in full and
  the log says why.
- A node applied after this one that replaces `sampler_calc_cond_batch_function` without chaining to it:
  the log says the accelerator was never called.

## Partial denoise

A partial `denoise` (img2img, a hires-fix pass) is planned on the steps that actually run, and whether it is
planned at all is judged by how dense those steps are, not by how many there are.

The reason for the 12-step minimum is spacing: with fewer steps over the whole schedule, neighbouring steps
are too far apart for one to stand in for another. A partial run is cut from a finer schedule. KSampler at
`denoise` 0.35 and 9 steps samples a 25-step schedule (`int(9 / 0.35)`) and runs its last 9 steps, so each
of those steps is as close to the next as in a full 25-step run. Counting steps alone would leave this pass
unaccelerated; counting density plans it like the end of a 25-step run.

The node finds the share of the schedule a run covers, its `coverage`, from the run's first sigma and the
model's own sigma curve (`percent_to_sigma` of its `model_sampling`), and plans the run when
`steps / coverage`, rounded to a whole number (KSampler's grid is `int(steps / denoise)`, a whole number of
steps), is at least 12 and `steps` is at least 6. Below 6 steps there is no room for a warmup, a
pattern and a full last step. A full run has coverage 1, so it is planned exactly as before. If the curve
cannot be read (it is missing, raises, or does not fall from start to end), coverage counts as 1 and the
run is judged by its step count; that never accelerates more than before.

| Run | Steps | Coverage | Equivalent | Plan with `balanced` | Forwards |
|---|---|---|---|---|---|
| full | 8 | 1 | 8 | `FFFFFFFF` | 16/16 |
| `denoise` 0.35 | 9 | 0.36 | 25 | `FFFCCFCCF` | 14/18 |
| `denoise` 0.6 | 15 | 0.6 | 25 | `FFFFFCCFCCFCCFF` | 24/30 |
| `denoise` 0.2 | 5 | 0.2 | 25 | `FFFFF` | 10/10 |

The warmup share and the pattern apply to the steps that run, as in a full run.

## Limits

- The presets were set for MOSS-Image. Other flow-matching models get the same plan, and the log says so.
- Steps marked `C` give the negative prompt less say, and steps marked `N` none, so a negative prompt that matters late in sampling
  (hands, fine detail) is weaker there. The warmup keeps the early steps, where composition is set, in full.
- The node does not check outputs for NaN; a bad step spreads to the steps that reuse it.

## Going back

Set `preset` to `off`, or bypass the node. Nothing is stored outside the sampling run.

## Tests

```sh
./check
COMFYUI_PATH=/path/to/ComfyUI MOSS_ACCEL_PYTHON=/path/to/comfy/venv/bin/python ./check
```

The plan tests need only Python. The ComfyUI tests build a small MOSS-Image model with random weights on the
CPU and sample with the real ComfyUI samplers; without `COMFYUI_PATH` they are skipped and say why.
