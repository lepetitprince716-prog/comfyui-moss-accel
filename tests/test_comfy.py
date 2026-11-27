"""The node inside a real ComfyUI checkout, on a small MOSS-Image model with random weights, on the CPU.

Set COMFYUI_PATH to a ComfyUI checkout and run with a Python that has its requirements; otherwise these
tests are skipped and the reason is printed.
"""

import asyncio
import importlib.util
import os
import pathlib
import sys
import unittest

PROJECT = pathlib.Path(__file__).resolve().parent.parent
COMFY = os.environ.get("COMFYUI_PATH", "")


def _skip_reason():
    if not COMFY:
        return "COMFYUI_PATH is not set"
    if not os.path.isdir(os.path.join(COMFY, "comfy")):
        return f"COMFYUI_PATH {COMFY} is not a ComfyUI checkout"
    if importlib.util.find_spec("torch") is None:
        return f"torch is not importable from {sys.executable}"
    return None


SKIP = _skip_reason()
if SKIP:
    print(f"test_comfy: skipped, {SKIP}", file=sys.stderr)
else:
    sys.path.insert(0, COMFY)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import torch

    import comfy.model_patcher
    import comfy.model_sampling
    import comfy.sample
    import comfy.samplers
    import comfy.supported_models
    import nodes

    asyncio.run(nodes.load_custom_node(str(PROJECT)))
    Node = nodes.NODE_CLASS_MAPPINGS.get("MOSSImageAccelerate")
    package = sys.modules[str(PROJECT).replace(".", "_x_")]
    plan = sys.modules[package.__name__ + ".plan"]
    runtime = sys.modules[package.__name__ + ".runtime"]


class ComfyRunsInCI(unittest.TestCase):
    """CI sets CI=true. Skipping the ComfyUI tests there would leave the runtime untested behind a green check."""

    @unittest.skipUnless(os.environ.get("CI") == "true", "only in CI")
    def test_the_comfyui_tests_are_not_skipped(self):
        self.assertIsNone(SKIP, f"CI must run the ComfyUI tests, but they were skipped: {SKIP}")


def make_model():
    config = comfy.supported_models.MOSSImage({"image_model": "moss_image", "dim": 64, "ffn_dim": 128, "num_heads": 4,
                                               "num_layers": 2, "text_dim": 32, "latent_channels": 128})
    config.set_inference_dtype(torch.float32, torch.float32)
    model = config.get_model({}, device="cpu")
    # Unit norms and fan-in scaled weights: at this size the negative prompt then moves the result by about 8%
    # (CFG 2.5 against CFG 1). With small weights everywhere it moved it by 5e-6, and every guidance test passed vacuously.
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.fill_(1.0)
            elif p.ndim == 1:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.02)
            else:
                p.copy_(torch.randn(p.shape, generator=generator) / p.shape[1] ** 0.5)
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)


class Recorder:
    """Every DiT forward, as (sigma, branches): 0 is the positive prompt, 1 the negative."""

    def __init__(self):
        self.calls = []

    def __call__(self, apply_model, args):
        self.calls.append((float(args["timestep"].flatten()[0]), tuple(args["cond_or_uncond"])))
        return apply_model(args["input"], args["timestep"], **args["c"])

    def steps(self):
        """Branches per evaluated sigma, in call order."""
        grouped = []
        for sigma, branches in self.calls:
            if grouped and grouped[-1][0] == sigma:
                grouped[-1][1].extend(branches)
            else:
                grouped.append((sigma, list(branches)))
        return [(sigma, tuple(sorted(b))) for sigma, b in grouped]


def expected_steps(p, guided=True):
    branches = {"F": (0, 1) if guided else (0,), "C": (0,), "S": None, "N": (0,)}
    return [branches[a.value] for a in p]


@unittest.skipIf(SKIP, SKIP or "")
class ComfyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = make_model()
        generator = torch.Generator().manual_seed(1)
        cls.cond = [[torch.randn(1, 7, 32, generator=generator), {}]]
        cls.neg = [[torch.randn(1, 3, 32, generator=generator), {}]]
        cls.latent = torch.zeros(1, 128, 4, 4)
        cls.noise = torch.randn(1, 128, 4, 4, generator=generator)

    def recorded(self, model=None):
        model = (model or self.base).clone()
        recorder = Recorder()
        model.set_model_unet_function_wrapper(recorder)
        return model, recorder

    def sample(self, model, steps, cfg=2.5, sampler="er_sde", denoise=1.0, latent=None):
        latent = self.latent if latent is None else latent
        return comfy.sample.sample(model, self.noise.clone(), steps, cfg, sampler, "simple", self.cond, self.neg, latent,
                                   denoise=denoise, seed=1, disable_pbar=True)

    def accelerate(self, model, preset, **custom):
        return Node.execute(model, preset, **custom).result[0]

    def assert_calls_follow(self, recorder, p, guided=True):
        got = [b for _, b in recorder.steps()]
        want = [b for b in expected_steps(p, guided) if b is not None]
        self.assertEqual(got, want)

    def test_the_model_responds_to_the_negative_prompt(self):
        guided, unguided = self.sample(self.base, 25, cfg=2.5), self.sample(self.base, 25, cfg=1.0)
        self.assertGreater(((guided - unguided).norm() / unguided.norm()).item(), 0.02)

    def test_loads_through_the_custom_node_loader(self):
        self.assertIsNotNone(Node)
        self.assertEqual(nodes.NODE_DISPLAY_NAME_MAPPINGS["MOSSImageAccelerate"], "MOSS-Image Accelerate")
        self.assertIn("MOSSRowW8A16CLIPLoader", nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("MOSSSlimCLIPLoader", nodes.NODE_CLASS_MAPPINGS)

    def test_off_is_bitwise_identical_to_no_node(self):
        plain = self.sample(self.base, 15)
        model, recorder = self.recorded()
        off = self.sample(self.accelerate(model, "off"), 15)
        self.assertTrue(torch.equal(plain, off))
        self.assertEqual(len(recorder.calls), 30)

    def test_off_after_an_accelerator_takes_it_out(self):
        plain = self.sample(self.base, 15)
        model, recorder = self.recorded()
        off = self.sample(self.accelerate(self.accelerate(model, "fast"), "off"), 15)
        self.assertTrue(torch.equal(plain, off))
        self.assertEqual(len(recorder.calls), 30)

    def test_forwards_follow_the_plan(self):
        for preset in ("light", "balanced", "fast"):
            for steps in (12, 15):
                with self.subTest(preset=preset, steps=steps):
                    model, recorder = self.recorded()
                    self.sample(self.accelerate(model, preset), steps)
                    p = plan.plan_steps(steps, plan.PRESETS[preset])
                    self.assert_calls_follow(recorder, p)
                    self.assertEqual(len(recorder.calls), plan.forward_count(p))

    def test_custom_inputs_reach_the_plan(self):
        model, recorder = self.recorded()
        self.sample(self.accelerate(model, "custom", warmup=0.5, pattern="cs", skip_mode="linear", cache_space="v"), 14)
        p = plan.plan_steps(14, plan.Settings.parse(0.5, "CS", "linear", "v"))
        self.assertEqual("".join(a.value for a in p), "FFFFFFFCSCSCCF")
        self.assert_calls_follow(recorder, p)

    def test_unguided_steps_run_the_positive_prompt_only(self):
        model, recorder = self.recorded()
        self.sample(self.accelerate(model, "custom", warmup=0.32, pattern="NNF"), 25)
        p = plan.plan_steps(25, plan.Settings.parse(0.32, "NNF"))
        self.assertEqual("".join(a.value for a in p), "FFFFFFFFNNFNNFNNFNNFNNFNF")
        self.assert_calls_follow(recorder, p)
        self.assertEqual(len(recorder.calls), plan.forward_count(p))

    def test_refine_runs_a_partial_denoise_pass_with_no_guidance_after_the_warmup(self):
        latent = torch.randn(1, 128, 4, 4, generator=torch.Generator().manual_seed(2))
        p = plan.plan_steps(9, plan.PRESETS["refine"], 0.36)
        self.assertEqual("".join(a.value for a in p), "FFNNNNNNF")
        model, recorder = self.recorded()
        with self.assertLogs("moss_image_accelerate", "INFO") as logs:
            self.sample(self.accelerate(model, "refine"), 9, denoise=0.35, latent=latent)
        self.assertEqual(len(recorder.calls), 12)
        self.assert_calls_follow(recorder, p)
        self.assertIn("preset refine, 9 steps (coverage 0.36, 25.00 equivalent steps), plan FFNNNNNNF", logs.output[-1])

    def test_custom_refuses_bad_input(self):
        with self.assertRaises(ValueError):
            self.accelerate(self.base, "custom", pattern="SSF")
        with self.assertRaises(ValueError):
            self.accelerate(self.base, "custom", cache_space="eps")

    def test_input_model_is_left_unchanged(self):
        before = dict(self.base.model_options)
        wrappers = {k: dict(v) for k, v in self.base.wrappers.items()}
        self.accelerate(self.base, "fast")
        self.assertEqual(dict(self.base.model_options), before)
        self.assertEqual({k: dict(v) for k, v in self.base.wrappers.items()}, wrappers)

    def test_two_samplings_are_identical(self):
        model, recorder = self.recorded()
        fast = self.accelerate(model, "fast")
        first = self.sample(fast, 15)
        first_calls = list(recorder.calls)
        recorder.calls.clear()
        second = self.sample(fast, 15)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(recorder.calls, first_calls)

    def test_accelerated_image_stays_close(self):
        plain = self.sample(self.base, 25)
        for preset in ("light", "balanced", "fast"):
            with self.subTest(preset=preset):
                out = self.sample(self.accelerate(self.base, preset), 25)
                self.assertFalse(torch.equal(plain, out))
                rel = ((out - plain).norm() / plain.norm()).item()
                self.assertLess(rel, 0.2)

    def sample_with(self, guider, model, steps=15):
        """A sampling through a guider, as SamplerCustomAdvanced runs it, on the simple scheduler."""
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", steps)
        return guider.sample(self.noise.clone(), self.latent, comfy.samplers.sampler_object("er_sde"), sigmas, seed=1)

    def real_guider(self, kind, model, cfg=None):
        from comfy_extras import nodes_custom_sampler

        if kind == "cfg":
            return nodes_custom_sampler.CFGGuider.execute(model, self.cond, self.neg, cfg).result[0]
        return nodes_custom_sampler.BasicGuider.execute(model, self.cond).result[0]

    def test_cfg_guider_above_1_is_planned(self):
        model, recorder = self.recorded()
        self.sample_with(self.real_guider("cfg", self.accelerate(model, "fast"), 2.5), model)
        p = plan.plan_steps(15, plan.PRESETS["fast"])
        self.assertEqual(len(recorder.calls), plan.forward_count(p))
        self.assert_calls_follow(recorder, p)

    def test_cfg_1_does_no_guidance_so_every_step_runs_in_full(self):
        # ComfyUI leaves the negative prompt out at CFG 1: nothing to save, and a distilled few-step recipe has
        # no steps to spare. Through KSampler and through CFGGuider alike, one forward per step, the image unchanged.
        plain = self.sample(self.base, 15, cfg=1.0)
        model, recorder = self.recorded()
        with self.assertLogs("moss_image_accelerate", "INFO") as logs:
            out = self.sample(self.accelerate(model, "fast"), 15, cfg=1.0)
        self.assertTrue(torch.equal(plain, out))
        self.assertEqual([b for _, b in recorder.steps()], [(0,)] * 15)
        self.assertIn("MOSS-Image Accelerate: guider CFGGuider has cfg 1: no classifier-free guidance to reuse, every step runs in full, inactive for this sampling",
                      logs.output[0])
        self.assertIn("inactive: guider CFGGuider has cfg 1: no classifier-free guidance to reuse", logs.output[-1])

        plain = self.sample_with(self.real_guider("cfg", self.base, 1.0), self.base)
        model, recorder = self.recorded()
        out = self.sample_with(self.real_guider("cfg", self.accelerate(model, "fast"), 1.0), model)
        self.assertTrue(torch.equal(plain, out))
        self.assertEqual([b for _, b in recorder.steps()], [(0,)] * 15)

    def test_basic_guider_does_no_guidance_so_every_step_runs_in_full(self):
        plain = self.sample_with(self.real_guider("basic", self.base), self.base)
        model, recorder = self.recorded()
        with self.assertLogs("moss_image_accelerate", "INFO") as logs:
            out = self.sample_with(self.real_guider("basic", self.accelerate(model, "fast")), model)
        self.assertTrue(torch.equal(plain, out))
        self.assertEqual([b for _, b in recorder.steps()], [(0,)] * 15)
        self.assertIn("guider Guider_Basic has cfg 1: no classifier-free guidance to reuse, every step runs in full, inactive for this sampling",
                      logs.output[0])

    def test_a_guider_without_a_cfg_does_no_guidance(self):
        class Bare:
            pass

        class Text:
            cfg = "x"

        for guider in (Bare(), Text()):
            with self.subTest(guider=type(guider).__name__):
                self.assertIn("does no classifier-free guidance", runtime._no_guidance(guider))
        self.assertIsNone(runtime._no_guidance(type("Guided", (), {"cfg": 2.5})()))
        self.assertIsNotNone(runtime._no_guidance(type("Guided", (), {"cfg": 1.0})()))

    def test_the_guard_names_the_guider_and_its_cfg(self):
        # Guider_DualCFG keeps the base class's cfg of 1.0 yet does use a negative, so the reason says what was read, not why.
        reason = runtime._no_guidance(type("Guider_DualCFG", (), {"cfg": 1.0})())
        self.assertEqual(reason, "guider Guider_DualCFG has cfg 1: no classifier-free guidance to reuse, every step runs in full")

    def test_the_cfg_guard_is_pinned_on_both_sides_of_1(self):
        # Below 1 ComfyUI still computes the negative (a guidance weight under 1 blends toward it), so it is planned.
        for cfg, active in ((0.5, True), (1.0, False), (1.0 + 1e-12, False), (1.01, True), (1.6, True), (2.5, True)):
            with self.subTest(cfg=cfg):
                reason = runtime._no_guidance(type("Guided", (), {"cfg": cfg})())
                self.assertEqual(reason is None, active)

    def test_a_run_without_a_negative_prompt_runs_every_step_but_the_skips(self):
        # A guider that hands the hook a positive prompt only (a third-party guider, or any path that drops the
        # negative) gets one forward per full or cond-only step, and on a skipped step the prediction extrapolated
        # from the last two evaluated steps (the fast preset skips by linear) with zeros for the negative.
        model_sampling = self.base.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, "simple", 25)
        x = torch.full((1, 1, 2, 2), 3.0)
        options = {"transformer_options": {"sample_sigmas": sigmas}}
        run = runtime.Run(plan.PRESETS["fast"], "test")
        computed = []

        def compute(args):
            self.assertIsNone(args["conds"][1])
            computed.append(torch.full_like(x, len(computed) + 1.0))
            return [computed[-1], torch.zeros_like(x)]

        outputs = [run.evaluate({"input": x, "sigma": sigmas[i : i + 1], "conds": [0, None], "model_options": options}, compute)
                   for i in range(25)]
        p = plan.plan_steps(25, plan.PRESETS["fast"])
        self.assertEqual("".join(a.value for a in p), "FFFFFFFFFCSFCSFCSFCSFCSFF")
        self.assertEqual(len(computed), 20)
        self.assertEqual((run.forwards, run.guided_calls), (20, 0))
        self.assertEqual((run.served[plan.Action.FULL], run.served[plan.Action.COND], run.served[plan.Action.SKIP]), (20, 0, 5))
        evaluated = []
        for i, (action, out) in enumerate(zip(p, outputs)):
            if action is plan.Action.SKIP:
                (first_sigma, first), (last_sigma, last) = evaluated[-2:]
                k = (float(sigmas[i]) - last_sigma) / (last_sigma - first_sigma)
                torch.testing.assert_close(out[0], last + k * (last - first))
                self.assertFalse(torch.equal(out[0], last))
                self.assertTrue(torch.equal(out[1], torch.zeros_like(x)))
            else:
                evaluated.append((float(sigmas[i]), out[0]))

    def test_cfg_1_with_no_skips_is_identical_to_no_node(self):
        plain = self.sample(self.base, 15, cfg=1.0)
        out = self.sample(self.accelerate(self.base, "balanced"), 15, cfg=1.0)
        self.assertTrue(torch.equal(plain, out))

    def test_short_runs_are_untouched(self):
        plain = self.sample(self.base, 8)
        model, recorder = self.recorded()
        out = self.sample(self.accelerate(model, "fast"), 8)
        self.assertTrue(torch.equal(plain, out))
        self.assertEqual(len(recorder.calls), 16)

    def test_heun_runs_every_evaluation_in_full(self):
        plain = self.sample(self.base, 12, sampler="heun")
        model, recorder = self.recorded()
        out = self.sample(self.accelerate(model, "fast"), 12, sampler="heun")
        self.assertTrue(torch.equal(plain, out))
        self.assertEqual(len(recorder.calls), 2 * (2 * 12 - 1))

    def test_truncated_img2img_schedule(self):
        latent = torch.randn(1, 128, 4, 4, generator=torch.Generator().manual_seed(2))
        model, recorder = self.recorded()
        self.sample(self.accelerate(model, "balanced"), 15, denoise=0.6, latent=latent)
        p = plan.plan_steps(15, plan.PRESETS["balanced"])
        self.assertLess(recorder.calls[0][0], 0.99)
        self.assert_calls_follow(recorder, p)

    def test_img2img_tail_is_planned_at_the_density_of_its_schedule(self):
        # KSampler at denoise 0.35 and 9 steps runs the last 9 steps of a 25-step schedule: as fine as a full
        # 25-step run, so it gets a plan although 9 is below the minimum step count.
        latent = torch.randn(1, 128, 4, 4, generator=torch.Generator().manual_seed(2))
        p = plan.plan_steps(9, plan.PRESETS["balanced"], 0.36)
        self.assertEqual(plan.forward_count(p), 14)
        model, recorder = self.recorded()
        with self.assertLogs("moss_image_accelerate", "INFO") as logs:
            self.sample(self.accelerate(model, "balanced"), 9, denoise=0.35, latent=latent)
        self.assertEqual(len(recorder.calls), 14)
        self.assert_calls_follow(recorder, p)
        self.assertIn("9 steps (coverage 0.36, 25.00 equivalent steps), plan FFFCCFCCF", logs.output[-1])

        model, recorder = self.recorded()
        self.sample(self.accelerate(model, "off"), 9, denoise=0.35, latent=latent)
        self.assertEqual(len(recorder.calls), 18)

    def test_tail_cut_from_a_12_step_schedule_is_planned(self):
        # KSampler at denoise 0.75 and 9 steps runs the last 9 steps of a 12-step schedule. The simple scheduler
        # rounds its cut points to a 1/1000 grid, so the coverage read from the curve is 0.750 and a hair off.
        latent = torch.randn(1, 128, 4, 4, generator=torch.Generator().manual_seed(2))
        model, recorder = self.recorded()
        with self.assertLogs("moss_image_accelerate", "INFO") as logs:
            self.sample(self.accelerate(model, "balanced"), 9, denoise=0.75, latent=latent)
        self.assertEqual(len(recorder.calls), 14)
        self.assertIn("9 steps (coverage 0.75, 12.00 equivalent steps), plan FFFCCFCCF", logs.output[-1])

    def test_tails_of_real_schedules_are_planned_by_the_grid_they_were_cut_from(self):
        # Sigmas as ComfyUI computes them for the MOSS-Image model_sampling (shift 5): a grid of n steps for a
        # tail of k steps, as KSampler cuts it. 1000 / n is a whole number for none of these n, so the cut points
        # sit off k / n by up to 1e-3 and only the grid decides: planned from 12 steps up, never below.
        model_sampling = self.base.get_model_object("model_sampling")
        self.assertEqual(model_sampling.shift, 5.0)
        curve = model_sampling.percent_to_sigma
        for scheduler in ("simple", "sgm_uniform"):
            for n in (12, 13, 11, 7, 9):
                sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, n)
                for k in range(1, n):
                    with self.subTest(scheduler=scheduler, grid=n, steps=k):
                        coverage = plan.coverage_of(float(sigmas[-(k + 1)]), curve)
                        self.assertAlmostEqual(coverage, k / n, delta=0.005)
                        got = plan.plan_steps(k, plan.PRESETS["balanced"], coverage)
                        self.assertEqual(any(a is not plan.Action.FULL for a in got), n >= 12 and k >= 6)

    def test_tails_of_every_uniform_grid_scheduler_are_planned_by_their_density(self):
        # The four schedulers that cut a grid of n evenly (karras, exponential, kl_optimal, beta and linear_quadratic
        # do not, and are not asserted). A tail of k steps cut from a grid of n steps is planned from n = 12 up. The only
        # tails below that which are planned are the two that start at half the schedule: normal at n = 11 and
        # ddim_uniform at n = 10 put their first sigma where a 12-step grid's 6-step tail does, so the node, which sees
        # only sigmas, cannot tell them apart and gives the same plan.
        halfway = {("normal", 11, 6), ("ddim_uniform", 10, 6)}
        model_sampling = self.base.get_model_object("model_sampling")
        curve = model_sampling.percent_to_sigma
        line = plan.MIN_STEPS - plan.EQUIVALENT_TOLERANCE
        for scheduler in ("simple", "sgm_uniform", "normal", "ddim_uniform"):
            for n in range(6, 31):
                sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, n)
                grid = len(sigmas) - 1  # ddim_uniform returns n + 1 steps for some n; the grid is what came back
                for k in range(6, grid):
                    with self.subTest(scheduler=scheduler, requested=n, grid=grid, steps=k):
                        coverage = plan.coverage_of(float(sigmas[-(k + 1)]), curve)
                        self.assertGreaterEqual(abs(plan.equivalent_steps(k, coverage) - line), 0.1)
                        got = plan.plan_steps(k, plan.PRESETS["balanced"], coverage)
                        self.assertEqual(any(a is not plan.Action.FULL for a in got),
                                         grid >= 12 or (scheduler, grid, k) in halfway)

    def test_six_step_tail_of_a_24_step_schedule_is_planned(self):
        # KSampler at denoise 0.25 and 6 steps: the last 6 steps of a 24-step schedule, the shortest run that is planned.
        latent = torch.randn(1, 128, 4, 4, generator=torch.Generator().manual_seed(2))
        model, recorder = self.recorded()
        with self.assertLogs("moss_image_accelerate", "INFO") as logs:
            self.sample(self.accelerate(model, "balanced"), 6, denoise=0.25, latent=latent)
        self.assertEqual(len(recorder.calls), 10)
        self.assertIn("6 steps (coverage 0.25, 24.00 equivalent steps), plan FFCCFF, forwards 10 of 12", logs.output[-1])

    def test_run_without_the_model_curve_counts_as_a_full_run(self):
        # A Run that cannot read percent_to_sigma plans by step count alone, so a 9-step tail stays all full.
        model_sampling = self.base.get_model_object("model_sampling")
        tail = comfy.samplers.calculate_sigmas(model_sampling, "simple", 25)[-10:]
        x = torch.zeros(1, 1, 2, 2)
        options = {"transformer_options": {"sample_sigmas": tail}}
        for sigma_at_percent, want in ((None, 18), (model_sampling.percent_to_sigma, 14)):
            with self.subTest(curve=sigma_at_percent is not None):
                run = runtime.Run(plan.PRESETS["balanced"], "test", sigma_at_percent)
                for i in range(9):
                    run.evaluate({"input": x, "sigma": tail[i : i + 1], "conds": [0, 1], "model_options": options},
                                 lambda args: [torch.zeros_like(x), torch.zeros_like(x)])
                self.assertEqual(run.forwards, want)

    def test_er_sde_offset_first_sigma_is_matched(self):
        model, recorder = self.recorded()
        self.sample(self.accelerate(model, "fast"), 15)
        first = recorder.calls[0][0]
        self.assertLess(first, 1.0)
        self.assertGreater(first, 0.999)
        self.assert_calls_follow(recorder, plan.plan_steps(15, plan.PRESETS["fast"]))

    def test_existing_hook_is_chained(self):
        seen = []

        def previous(args):
            seen.append(tuple(c is not None for c in args["conds"]))
            return comfy.samplers.calc_cond_batch(args["model"], args["conds"], args["input"], args["sigma"], args["model_options"])

        alone = self.sample(self.accelerate(self.base, "fast"), 15)
        hooked = self.base.clone()
        hooked.set_model_sampler_calc_cond_batch_function(previous)
        chained = self.accelerate(hooked, "fast")
        out = self.sample(chained, 15)
        self.assertTrue(torch.equal(alone, out))
        p = "".join(a.value for a in plan.plan_steps(15, plan.PRESETS["fast"]))
        self.assertEqual(seen, [(True, a == "F") for a in p if a != "S"])

        seen.clear()
        twice = self.sample(self.accelerate(chained, "fast"), 15)
        self.assertTrue(torch.equal(alone, twice))
        self.assertEqual(len(seen), len(p) - p.count("S"))

    def test_linear_skip_lands_on_the_line_through_the_last_two_steps(self):
        # A model whose x0 is linear in sigma: extrapolating the last two evaluated steps over sigma has to land
        # on that line exactly, for the positive and the negative branch alike.
        settings = plan.Settings.parse(0.2, "SF", "linear", "x0")
        schedule = torch.tensor([1.0 - 0.05 * i for i in range(20)] + [0.0])
        p = plan.plan_steps(20, settings)
        self.assertGreater(p.count(plan.Action.SKIP), 3)
        run = runtime.Run(settings, "test")
        x = torch.zeros(1, 1, 2, 2)
        options = {"transformer_options": {"sample_sigmas": schedule}}
        computed = []

        def lines(sigma):
            return [torch.full_like(x, 1.0 + 2.0 * sigma), torch.full_like(x, 0.5 - sigma)]

        def compute(args):
            computed.append(args["conds"])
            return lines(float(args["sigma"][0]))

        for i, action in enumerate(p):
            sigma = schedule[i : i + 1]
            before = len(computed)
            out = run.evaluate({"input": x, "sigma": sigma, "conds": [0, 1], "model_options": options}, compute)
            if action is plan.Action.SKIP:
                with self.subTest(step=i):
                    self.assertEqual(len(computed), before)
                    want = lines(float(sigma[0]))
                    torch.testing.assert_close(out[0], want[0])
                    torch.testing.assert_close(out[1], want[1])
        self.assertEqual(run.served[plan.Action.SKIP], p.count(plan.Action.SKIP))

    def test_foreign_hook_between_two_accelerators_keeps_the_plan(self):
        # The foreign hook wraps the first accelerator and the second wraps the hook, so the first is reached from
        # inside the second's evaluation. It has to pass the call through, or every step runs twice the plan.
        seen = []

        def wrap(inner):
            def hook(args):
                seen.append(args["sigma"])
                return inner(args)

            return hook

        model, recorder = self.recorded()
        first = self.accelerate(model, "fast")
        hooked = first.clone()
        hooked.set_model_sampler_calc_cond_batch_function(wrap(hooked.model_options["sampler_calc_cond_batch_function"]))
        out = self.sample(self.accelerate(hooked, "fast"), 15)
        p = plan.plan_steps(15, plan.PRESETS["fast"])
        self.assertEqual(plan.forward_count(p), 22)
        self.assertEqual(len(recorder.calls), plan.forward_count(p))
        self.assert_calls_follow(recorder, p)
        self.assertEqual(len(seen), len(p) - p.count(plan.Action.SKIP))
        self.assertTrue(torch.equal(out, self.sample(self.accelerate(self.base, "fast"), 15)))

    def test_non_flow_model_is_untouched(self):
        class EpsSampling(comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.EPS):
            pass

        eps = self.base.clone()
        eps.add_object_patch("model_sampling", EpsSampling())
        plain = self.sample(eps, 15)
        model, recorder = self.recorded(eps)
        out = self.sample(self.accelerate(model, "fast"), 15)
        self.assertTrue(torch.equal(plain, out))
        self.assertEqual(len(recorder.calls), 30)

    def _combined(self, settings, steps):
        # What the sampler combines at each step, read after this node: (sigma, input, [cond, uncond]) per step.
        seen = []

        def record(args):
            seen.append((float(args["sigma"].flatten()[0]), args["input"].clone(), [t.clone() for t in args["conds_out"]]))
            return args["conds_out"]

        model = self.base.clone()
        model.set_model_sampler_pre_cfg_function(record)
        self.sample(runtime.apply(model, "test", settings), steps)
        return seen

    def test_reused_guidance_and_held_steps_are_what_the_cache_space_says(self):
        # C steps rebuild the negative branch from the last full step's difference, S steps repeat the last
        # evaluated step, in x0 or rescaled for v. Hold is set here: the fast preset itself skips by linear.
        from dataclasses import replace

        p = "".join(a.value for a in plan.plan_steps(15, plan.PRESETS["fast"]))
        for space in ("x0", "v"):
            with self.subTest(space=space):
                settings = replace(plan.PRESETS["fast"], skip_mode=plan.SkipMode.HOLD, cache_space=plan.CacheSpace(space))
                seen = self._combined(settings, 15)
                self.assertEqual(len(seen), len(p))
                full = last = None
                for action, (sigma, x, (cond, uncond)) in zip(p, seen):
                    if action == "F":
                        full = last = (sigma, x, cond, uncond)
                        continue
                    if action == "C":
                        f_sigma, _, f_cond, f_uncond = full
                        scale = 1.0 if space == "x0" else sigma / f_sigma
                        torch.testing.assert_close(uncond, cond - scale * (f_cond - f_uncond), rtol=1e-5, atol=1e-5)
                        last = (sigma, x, cond, uncond)
                        continue
                    l_sigma, l_x, l_cond, l_uncond = last
                    for got, held in ((cond, l_cond), (uncond, l_uncond)):
                        want = held if space == "x0" else x - sigma * (l_x - held) / l_sigma
                        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def test_skipped_steps_of_the_fast_preset_land_on_the_line_through_the_last_two_evaluated_steps(self):
        # The fast preset as shipped (linear, x0), and the same in v: an S step extrapolates both branches over
        # sigma from the last two F or C steps, taken in the cache space and turned back into a prediction.
        from dataclasses import replace

        p = "".join(a.value for a in plan.plan_steps(15, plan.PRESETS["fast"]))
        self.assertIn("S", p)
        for space in ("x0", "v"):
            with self.subTest(space=space):
                seen = self._combined(replace(plan.PRESETS["fast"], cache_space=plan.CacheSpace(space)), 15)
                self.assertEqual(len(seen), len(p))

                def to_space(out, x, sigma):
                    return out if space == "x0" else (x - out) / sigma

                def from_space(value, x, sigma):
                    return value if space == "x0" else x - sigma * value

                evaluated = []
                off_hold = 0
                for action, (sigma, x, outs) in zip(p, seen):
                    if action != "S":
                        evaluated.append((sigma, [to_space(out, x, sigma) for out in outs]))
                        continue
                    self.assertGreaterEqual(len(evaluated), 2)
                    (f_sigma, first), (l_sigma, last) = evaluated[-2:]
                    k = (sigma - l_sigma) / (l_sigma - f_sigma)
                    for got, f_q, l_q in zip(outs, first, last):
                        torch.testing.assert_close(got, from_space(l_q + k * (l_q - f_q), x, sigma), rtol=1e-5, atol=1e-5)
                        if not torch.allclose(got, from_space(l_q, x, sigma), rtol=1e-5, atol=1e-5):
                            off_hold += 1
                self.assertGreater(off_hold, 0)

    def _space(self, settings, space, steps):
        from dataclasses import replace

        model = runtime.apply(self.base, "test", replace(settings, cache_space=plan.CacheSpace(space)))
        return self.sample(model, steps, sampler="euler")

    def test_hold_is_the_same_in_x0_and_v_under_euler(self):
        # Only F and S steps: plan FFFFFSFSFSFF. The derivation x_{i+1} - s_{i+1} v_i = x_i - s_i v_i covers this case.
        settings = plan.Settings.parse(0.4, "SF")
        self.assertEqual("".join(a.value for a in plan.plan_steps(12, settings)), "FFFFFSFSFSFF")
        x0, v = self._space(settings, "x0", 12), self._space(settings, "v", 12)
        torch.testing.assert_close(x0, v, rtol=1e-5, atol=1e-5)


@unittest.skipIf(SKIP, SKIP or "")
class UnguidedStepTest(unittest.TestCase):
    """N steps through Run.evaluate with a stand-in for the model call, whose outputs depend on sigma."""

    STEPS = 25

    def setUp(self):
        self.schedule = torch.tensor([1.0 - 0.04 * i for i in range(self.STEPS)] + [0.0])
        self.x = torch.zeros(1, 1, 2, 2)
        self.options = {"transformer_options": {"sample_sigmas": self.schedule}}
        self.calls = []

    def lines(self, sigma):
        """cond and uncond of the stand-in model: the guidance, cond minus uncond, is 0.5 + 3 sigma and differs at every step."""
        return [torch.full_like(self.x, 1.0 + 2.0 * sigma), torch.full_like(self.x, 0.5 - sigma)]

    def compute(self, args):
        self.calls.append(args["conds"])
        return self.lines(float(args["sigma"][0]))

    def drive(self, pattern):
        settings = plan.Settings.parse(0.32, pattern)
        run = runtime.Run(settings, "test")
        p = plan.plan_steps(self.STEPS, settings)
        outs = []
        for i in range(self.STEPS):
            args = {"input": self.x, "sigma": self.schedule[i : i + 1], "conds": [0, 1], "model_options": self.options}
            outs.append(run.evaluate(args, self.compute))
        return run, p, outs

    def test_the_guided_output_of_an_n_step_is_the_positive_prediction(self):
        run, p, outs = self.drive("NNF")
        n_steps = [i for i, a in enumerate(p) if a is plan.Action.UNGUIDED]
        self.assertEqual(len(n_steps), 11)
        for i in n_steps:
            with self.subTest(step=i):
                cond, uncond = outs[i]
                torch.testing.assert_close(cond, self.lines(float(self.schedule[i]))[0])
                torch.testing.assert_close(uncond, cond)
                self.assertIsNot(uncond, cond)
                # What the sampler's CFG formula makes of it, at the scale of the other tests.
                torch.testing.assert_close(uncond + 2.5 * (cond - uncond), cond)

    def test_an_n_step_asks_the_model_for_the_positive_prompt_only(self):
        _, p, _ = self.drive("NNF")
        self.assertEqual([c[1] is None for c in self.calls], [a is not plan.Action.FULL for a in p])

    def test_an_n_step_leaves_the_held_guidance_and_the_history_alone(self):
        settings = plan.Settings.parse(0.32, "NNF")
        run = runtime.Run(settings, "test")
        p = plan.plan_steps(self.STEPS, settings)
        for i, action in enumerate(p):
            before = (run.guidance, tuple(run.history))
            args = {"input": self.x, "sigma": self.schedule[i : i + 1], "conds": [0, 1], "model_options": self.options}
            run.evaluate(args, self.compute)
            if action is plan.Action.UNGUIDED:
                with self.subTest(step=i):
                    self.assertIs(run.guidance, before[0])
                    self.assertEqual(tuple(run.history), before[1])
                    self.assertIsNotNone(run.guidance)

    def test_a_c_step_after_n_steps_reuses_the_guidance_of_the_last_full_step(self):
        run, p, outs = self.drive("NNC")
        last_full = None
        checked = 0
        for i, action in enumerate(p):
            sigma = float(self.schedule[i])
            if action is plan.Action.FULL:
                last_full = sigma
            elif action is plan.Action.COND:
                cond, uncond = outs[i]
                torch.testing.assert_close(cond - uncond, torch.full_like(self.x, 0.5 + 3.0 * last_full))
                checked += 1
        self.assertGreater(checked, 3)

    def test_forward_counts_include_one_per_n_step(self):
        run, p, _ = self.drive("NNF")
        self.assertEqual(run.forwards, plan.forward_count(p))
        self.assertEqual(run.forwards, 39)
        self.assertEqual(len(self.calls), 25)
        self.assertEqual((run.served[plan.Action.FULL], run.served[plan.Action.UNGUIDED]), (14, 11))
        self.assertEqual(run.guided_calls, 25)

    def test_the_log_line_counts_n_steps_only_when_the_plan_has_some(self):
        with_n, _, _ = self.drive("NNF")
        self.assertTrue(with_n.summary().endswith("served F14 C0 S0 N11"), with_n.summary())
        self.assertIn("forwards 39 of 50 (plan 39)", with_n.summary())
        without_n, _, _ = self.drive("CF")
        self.assertRegex(without_n.summary(), r"served F\d+ C\d+ S\d+$")
        self.assertNotIn(" N", without_n.summary().split("served")[1])

    def test_fast_50_runs_exact_48_forwards_and_linear_two_step_skips(self):
        steps = 50
        schedule = torch.tensor([1.0 - 0.02 * i for i in range(steps)] + [0.0])
        options = {"transformer_options": {"sample_sigmas": schedule}}
        run = runtime.Run(plan.PRESETS["fast_50"], "fast_50")
        p = plan.plan_steps(steps, plan.PRESETS["fast_50"])
        self.assertEqual("".join(a.value for a in p), "FFFFFFFFFFFFSSCSSFSSCSSFSSCSSFSSCSSFSCSCSCSFCNNNNF")
        outs = []
        for i in range(steps):
            args = {"input": self.x, "sigma": schedule[i : i + 1], "conds": [0, 1], "model_options": options}
            outs.append(run.evaluate(args, self.compute))
        self.assertEqual(run.forwards, 48)
        self.assertIn("forwards 48 of 100 (plan 48), served F18 C8 S20 N4", run.summary())
        # Two consecutive skips (steps 12 and 13) extrapolate linearly from evaluated steps 10 and 11;
        # self.lines(sigma) is affine in sigma, so linear extrapolation recovers cond & uncond exactly.
        for skip_i in (12, 13):
            want_c, want_u = self.lines(float(schedule[skip_i]))
            torch.testing.assert_close(outs[skip_i][0], want_c, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(outs[skip_i][1], want_u, rtol=1e-5, atol=1e-5)
        # Final polish tail N steps (45..48) return positive-only predictions with cond == uncond.
        for n_i in (45, 46, 47, 48):
            torch.testing.assert_close(outs[n_i][0], outs[n_i][1], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
