"""Step plans, schedules and settings. Standard library only: runs without torch or ComfyUI."""

import importlib.util
import math
import pathlib
import sys
import unittest

_PATH = pathlib.Path(__file__).resolve().parent.parent / "plan.py"
_spec = importlib.util.spec_from_file_location("moss_accel_plan", _PATH)
plan = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = plan
_spec.loader.exec_module(plan)

A = plan.Action


def text(actions):
    return "".join(a.value for a in actions)


def flow_schedule(steps, shift=5.0):
    """Shifted flow sigmas like ModelSamplingAuraFlow(5) with the simple scheduler, ending at 0."""
    ts = [1.0 - i / steps for i in range(steps)]
    return [shift * t / (1 + (shift - 1) * t) for t in ts] + [0.0]


# Reviewed by hand against the rules in plan_steps: warmup head full, last step full,
# no skip on the second-to-last step, the pattern repeating in between.
EXPECTED = {
    ("light", 25): "FFFFFFFFCFCFCFCFCFCFCFCFF",
    ("balanced", 25): "FFFFFFFFCCFCCFCCFCCFCCFCF",
    ("fast", 25): "FFFFFFFFFCSFCSFCSFCSFCSFF",
    ("fast", 20): "FFFFFFFFCSFCSFCSFCCF",
    ("fast", 12): "FFFFFCSFCSFF",
    ("balanced", 15): "FFFFFCCFCCFCCFF",
    # Made for a second, partial-denoise pass; on a full run it would leave out the guidance that settles the content.
    ("refine", 25): "FFFFFNNNNNNNNNNNNNNNNNNNF",
}
STEPS = (8, 15, 20, 25, 30, 50)


class PlanTest(unittest.TestCase):
    def test_pinned_plans(self):
        for (name, steps), expected in EXPECTED.items():
            with self.subTest(preset=name, steps=steps):
                self.assertEqual(text(plan.plan_steps(steps, plan.PRESETS[name])), expected)

    def test_rules_hold_for_every_preset_and_step_count(self):
        for name in ("light", "balanced", "fast", "refine"):
            settings = plan.PRESETS[name]
            for steps in range(1, 81):
                with self.subTest(preset=name, steps=steps):
                    p = text(plan.plan_steps(steps, settings))
                    self.assertEqual(len(p), steps)
                    if steps < plan.MIN_STEPS:
                        self.assertEqual(p, "F" * steps)
                        continue
                    head = max(2, math.ceil(settings.warmup * steps - 1e-9))
                    self.assertEqual(p[:head], "F" * head)
                    self.assertEqual(p[-1], "F")
                    self.assertNotEqual(p[-2], "S")
                    self.assertNotIn("SS", p)

    def test_savings_at_25_steps_match_the_targets(self):
        targets = {"light": 0.16, "balanced": 0.22, "fast": 0.30}
        for name, target in targets.items():
            with self.subTest(preset=name):
                saved = plan.saved_fraction(plan.plan_steps(25, plan.PRESETS[name]))
                self.assertAlmostEqual(saved, target, delta=0.005)

    def test_savings_stay_ordered_across_step_counts(self):
        for steps in STEPS:
            light, balanced, fast = (plan.saved_fraction(plan.plan_steps(steps, plan.PRESETS[n]))
                                     for n in ("light", "balanced", "fast"))
            with self.subTest(steps=steps):
                if steps < plan.MIN_STEPS:
                    self.assertEqual((light, balanced, fast), (0.0, 0.0, 0.0))
                else:
                    self.assertLess(light, balanced)
                    self.assertLess(balanced, fast)

    def test_off_and_short_runs_are_all_full(self):
        self.assertEqual(text(plan.plan_steps(25, plan.PRESETS["off"])), "F" * 25)
        self.assertEqual(text(plan.plan_steps(8, plan.PRESETS["fast"])), "F" * 8)
        self.assertEqual(text(plan.plan_steps(11, plan.PRESETS["fast"])), "F" * 11)

    def test_forward_counts(self):
        p = plan.plan_steps(25, plan.PRESETS["fast"])
        self.assertEqual(plan.forward_count(p), 35)
        self.assertEqual(plan.forward_count(p, guided=False), 25 - text(p).count("S"))


def flow_percent_to_sigma(shift=5.0):
    """ModelSamplingDiscreteFlow.percent_to_sigma: the same shifted curve the simple scheduler samples."""

    def sigma_at_percent(percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        t = 1.0 - percent
        return shift * t / (1 + (shift - 1) * t)

    return sigma_at_percent


def ksampler_tail(steps, denoise, shift=5.0):
    """The sigmas KSampler runs at a partial denoise: the last steps + 1 of a schedule of int(steps / denoise)."""
    return flow_schedule(int(steps / denoise), shift)[-(steps + 1):]


def planned(steps, name, schedule):
    return text(plan.plan_steps(steps, plan.PRESETS[name], plan.coverage_of(schedule[0], flow_percent_to_sigma())))


class TailTest(unittest.TestCase):
    """A partial-denoise run is planned by how dense its steps are on the whole schedule, not by its step count."""

    def test_nine_step_tail_of_a_25_step_schedule_is_planned(self):
        # KSampler at denoise 0.35 and 9 steps: the last 10 sigmas of the 25-step schedule.
        tail = ksampler_tail(9, 0.35)
        self.assertEqual(tail, flow_schedule(25)[-10:])
        self.assertAlmostEqual(plan.coverage_of(tail[0], flow_percent_to_sigma()), 0.36, places=9)
        expected = {"light": "FFFCFCFCF", "balanced": "FFFCCFCCF", "fast": "FFFFCSFCF"}
        for name, want in expected.items():
            with self.subTest(preset=name):
                self.assertEqual(planned(9, name, tail), want)
        self.assertEqual(plan.forward_count(plan.plan_steps(9, plan.PRESETS["balanced"], 0.36)), 14)

    def test_refine_plans_the_nine_step_tail_with_no_guidance_after_the_warmup(self):
        # The second pass of the example workflow: 9 steps at denoise 0.35, lanczos-upscaled from a finished base image.
        tail = ksampler_tail(9, 0.35)
        self.assertEqual(planned(9, "refine", tail), "FFNNNNNNF")
        p = plan.plan_steps(9, plan.PRESETS["refine"], 0.36)
        self.assertEqual(plan.forward_count(p), 12)
        self.assertEqual(plan.forward_count(p, guided=False), 9)

    def test_refine_is_unguided_after_a_warmup_of_a_fifth(self):
        settings = plan.PRESETS["refine"]
        self.assertEqual(settings.warmup, 0.2)
        self.assertEqual(settings.pattern, (A.UNGUIDED,))
        self.assertEqual(settings, plan.Settings.parse(0.2, "N"))

    def test_refine_leaves_a_pass_too_coarse_for_reuse_untouched(self):
        self.assertEqual(text(plan.plan_steps(8, plan.PRESETS["refine"])), "F" * 8)
        self.assertEqual(planned(5, "refine", ksampler_tail(5, 0.2)), "FFFFF")

    def test_complete_eight_step_run_is_not_planned(self):
        schedule = flow_schedule(8)
        self.assertEqual(plan.coverage_of(schedule[0], flow_percent_to_sigma()), 1.0)
        for name in ("light", "balanced", "fast"):
            self.assertEqual(planned(8, name, schedule), "F" * 8)

    def test_fifteen_step_tail_at_denoise_06_is_planned(self):
        tail = ksampler_tail(15, 0.6)
        self.assertAlmostEqual(plan.coverage_of(tail[0], flow_percent_to_sigma()), 0.6, places=9)
        for name in ("light", "balanced", "fast"):
            with self.subTest(preset=name):
                got = planned(15, name, tail)
                self.assertEqual(got, text(plan.plan_steps(15, plan.PRESETS[name])))
                self.assertNotEqual(got, "F" * 15)

    def test_five_step_tail_is_not_planned(self):
        # Dense enough (5 / 0.2 = 25 equivalent steps) but too short for a head, a pattern and a full last step.
        tail = ksampler_tail(5, 0.2)
        self.assertAlmostEqual(plan.coverage_of(tail[0], flow_percent_to_sigma()), 0.2, places=9)
        for name in ("light", "balanced", "fast"):
            self.assertEqual(planned(5, name, tail), "F" * 5)

    def test_coverage_only_gates_the_plan(self):
        for name in ("light", "balanced", "fast"):
            settings = plan.PRESETS[name]
            for steps in range(1, 41):
                for coverage in (0.05, 0.2, 0.36, 0.5, 0.75, 0.99, 1.0):
                    with self.subTest(preset=name, steps=steps, coverage=coverage):
                        got = plan.plan_steps(steps, settings, coverage)
                        if steps < plan.MIN_PLAN_STEPS or steps / coverage < plan.MIN_STEPS:
                            self.assertEqual(text(got), "F" * steps)
                            continue
                        head = max(2, math.ceil(settings.warmup * steps - 1e-9))
                        p = text(got)
                        self.assertEqual(len(p), steps)
                        self.assertEqual(p[:head], "F" * head)
                        self.assertEqual(p[-1], "F")
                        self.assertNotEqual(p[-2], "S")
                        self.assertNotIn("SS", p)
                        if steps >= plan.MIN_STEPS:
                            self.assertEqual(got, plan.plan_steps(steps, settings))

    def test_plan_refuses_a_coverage_outside_0_to_1(self):
        for coverage in (0.0, -0.5, 1.5, float("nan"), float("inf")):
            with self.subTest(coverage=coverage):
                with self.assertRaises(ValueError):
                    plan.plan_steps(25, plan.PRESETS["fast"], coverage)

    def test_coverage_is_exact_along_the_grid(self):
        for n in (10, 25, 50, 100):
            schedule = flow_schedule(n)
            for k in range(1, n):
                with self.subTest(n=n, k=k):
                    self.assertAlmostEqual(plan.coverage_of(schedule[k], flow_percent_to_sigma()), 1 - k / n, places=9)

    def test_coverage_falls_back_to_full(self):
        def raises(_):
            raise RuntimeError("no model_sampling")

        def bump(p):
            return 2.0 if 0.4 < p < 0.6 else 1.0 - p

        def increasing(p):
            return p

        def nan(p):
            return float("nan") if 0 < p < 1 else 1.0 - p

        def flat(p):
            return 0.5

        curve = flow_percent_to_sigma()
        cases = {
            "raises": (0.5, raises),
            "increasing": (0.5, increasing),
            "bump inside": (0.3, bump),
            "nan inside": (0.5, nan),
            "flat": (0.5, flat),
            "not a number": ("x", curve),
            "nan sigma": (float("nan"), curve),
            "above the start": (1.5, curve),
            "at the start": (1.0, curve),
            "at the end": (0.0, curve),
        }
        for name, (sigma, function) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(plan.coverage_of(sigma, function), 1.0)

    def test_coverage_reads_tensor_like_values(self):
        class Value:
            def __init__(self, v):
                self.v = v

            def __float__(self):
                return self.v

        curve = flow_percent_to_sigma()
        sigma = flow_schedule(25)[16]
        got = plan.coverage_of(Value(sigma), lambda p: Value(curve(p)))
        self.assertAlmostEqual(got, 0.36, places=9)


class EquivalentStepsTest(unittest.TestCase):
    """KSampler's partial-denoise grid is a whole number of steps, but the measured coverage is a hair off it."""

    def test_a_coverage_a_hair_off_its_grid_still_counts_as_the_grid(self):
        # 7 steps cut from a 12-step schedule: ComfyUI's simple scheduler gives coverage 0.584 (7/12 = 0.5833...),
        # which divides to 11.986.
        self.assertAlmostEqual(plan.equivalent_steps(7, 0.584), 11.9863, places=3)
        self.assertAlmostEqual(plan.equivalent_steps(11, 0.917), 11.9956, places=3)
        self.assertEqual(text(plan.plan_steps(7, plan.PRESETS["balanced"], 0.584)), "FFFCCFF")
        self.assertEqual(text(plan.plan_steps(7, plan.PRESETS["fast"], 0.584)), "FFFCSFF")
        self.assertEqual(text(plan.plan_steps(11, plan.PRESETS["balanced"], 0.917)), "FFFFCCFCCFF")

    def test_a_grid_below_twelve_stays_unplanned(self):
        # 7 steps at coverage 0.62 is an 11-step grid.
        self.assertAlmostEqual(plan.equivalent_steps(7, 0.62), 11.2903, places=3)
        for name in ("light", "balanced", "fast"):
            self.assertEqual(text(plan.plan_steps(7, plan.PRESETS[name], 0.62)), "F" * 7)


class ToleranceTest(unittest.TestCase):
    """The gate is 12 less a fixed margin, pinned from both sides with literal plans."""

    def test_the_line_is_at_11_75(self):
        self.assertEqual(plan.MIN_STEPS - plan.EQUIVALENT_TOLERANCE, 11.75)

    def test_just_above_the_line_is_planned(self):
        for density in (11.76, 11.94, 12.0):
            with self.subTest(density=density):
                self.assertEqual(text(plan.plan_steps(7, plan.PRESETS["balanced"], 7 / density)), "FFFCCFF")

    def test_just_below_the_line_is_not(self):
        for density in (11.74, 11.63, 11.0):
            with self.subTest(density=density):
                self.assertEqual(text(plan.plan_steps(7, plan.PRESETS["balanced"], 7 / density)), "FFFFFFF")


class StepFloorTest(unittest.TestCase):
    """The 6-step floor, pinned from both sides with literal plans."""

    def test_six_steps_of_a_dense_tail_are_planned(self):
        # Coverage 0.25 is a 24-step schedule.
        want = {"light": "FFCFCF", "balanced": "FFCCFF", "fast": "FFFCCF"}
        for name, plan_text in want.items():
            with self.subTest(preset=name):
                self.assertEqual(text(plan.plan_steps(6, plan.PRESETS[name], 0.25)), plan_text)

    def test_five_steps_of_a_dense_tail_are_not(self):
        for name in ("light", "balanced", "fast"):
            with self.subTest(preset=name):
                self.assertEqual(text(plan.plan_steps(5, plan.PRESETS[name], 0.2)), "FFFFF")
                self.assertEqual(text(plan.plan_steps(5, plan.PRESETS[name], 0.05)), "FFFFF")


class ScheduleTest(unittest.TestCase):
    def test_parse_rejects_bad_schedules(self):
        self.assertIsNone(plan.parse_schedule([1.0]))
        self.assertIsNone(plan.parse_schedule([1.0, float("nan"), 0.0]))
        self.assertIsNone(plan.parse_schedule([1.0, float("inf"), 0.0]))
        self.assertIsNone(plan.parse_schedule([1.0, 0.5, 0.5, 0.0]))
        self.assertIsNone(plan.parse_schedule([0.5, 1.0, 0.0]))
        self.assertEqual(plan.parse_schedule([1, 0.5, 0]), (1.0, 0.5, 0.0))

    def test_locate_step_matches_each_sigma(self):
        schedule = flow_schedule(25)
        for i, sigma in enumerate(schedule[:-1]):
            self.assertEqual(plan.locate_step(sigma, schedule), i)

    def test_er_sde_first_sigma_offset_still_matches(self):
        schedule = flow_schedule(25)
        self.assertEqual(plan.locate_step(1.0 - 2e-5, schedule), 0)

    def test_midpoints_do_not_match(self):
        schedule = flow_schedule(25)
        mid = (schedule[10] + schedule[11]) / 2
        self.assertIsNone(plan.locate_step(mid, schedule))
        self.assertIsNone(plan.locate_step(float("nan"), schedule))

    def test_cursor_follows_a_single_evaluation_sampler(self):
        schedule = flow_schedule(25)
        p = plan.plan_steps(25, plan.PRESETS["fast"])
        cursor = plan.Cursor(schedule, p)
        served = [cursor.advance(s) for s in schedule[:-1]]
        self.assertEqual(served, list(p))
        self.assertFalse(cursor.irregular)

    def test_cursor_goes_full_after_a_repeated_sigma(self):
        # Heun evaluates sigma_i and sigma_{i+1} per step: the third call repeats sigma_1.
        schedule = flow_schedule(25)
        p = plan.plan_steps(25, plan.PRESETS["fast"])
        cursor = plan.Cursor(schedule, p)
        calls = [schedule[0], schedule[1]]
        for i in range(1, 24):
            calls += [schedule[i], schedule[i + 1]]
        calls.append(schedule[24])
        served = [cursor.advance(s) for s in calls]
        self.assertTrue(cursor.irregular)
        self.assertEqual(cursor.irregular_at, 3)
        self.assertTrue(all(a is A.FULL for a in served))

    def test_cursor_goes_full_after_an_off_schedule_sigma(self):
        schedule = flow_schedule(25)
        p = plan.plan_steps(25, plan.PRESETS["fast"])
        cursor = plan.Cursor(schedule, p)
        for s in schedule[:9]:
            cursor.advance(s)
        self.assertIs(cursor.advance((schedule[9] + schedule[10]) / 2), A.FULL)
        self.assertTrue(all(cursor.advance(s) is A.FULL for s in schedule[9:-1]))

    def test_truncated_img2img_schedule_plans_on_its_own_steps(self):
        # KSampler with denoise 0.6 at 25 steps keeps the last 26 sigmas of a 41-step schedule.
        full = flow_schedule(41)
        schedule = plan.parse_schedule(full[-26:])
        self.assertIsNotNone(schedule)
        steps = len(schedule) - 1
        self.assertEqual(steps, 25)
        p = plan.plan_steps(steps, plan.PRESETS["balanced"])
        cursor = plan.Cursor(schedule, p)
        self.assertEqual([cursor.advance(s) for s in schedule[:-1]], list(p))
        self.assertFalse(cursor.irregular)

    def test_cursor_refuses_a_plan_of_the_wrong_length(self):
        with self.assertRaises(ValueError):
            plan.Cursor(flow_schedule(10), plan.plan_steps(9, None))


class SettingsTest(unittest.TestCase):
    def test_defaults(self):
        s = plan.Settings.parse(0.3, "csf")
        self.assertEqual(text(s.pattern), "CSF")
        self.assertIs(s.skip_mode, plan.SkipMode.HOLD)
        self.assertIs(s.cache_space, plan.CacheSpace.X0)
        for name in ("light", "balanced", "fast"):
            self.assertIs(plan.PRESETS[name].cache_space, plan.CacheSpace.X0)

    def test_fast_preset_skips_by_extrapolation(self):
        self.assertIs(plan.PRESETS["fast"].skip_mode, plan.SkipMode.LINEAR)
        for name in ("light", "balanced"):
            self.assertIs(plan.PRESETS[name].skip_mode, plan.SkipMode.HOLD)

    def test_all_options_parse(self):
        s = plan.Settings.parse(0.5, "FS", "linear", "v")
        self.assertIs(s.skip_mode, plan.SkipMode.LINEAR)
        self.assertIs(s.cache_space, plan.CacheSpace.V)

    def test_bad_input_is_refused(self):
        bad = [
            (0.1, "CF", "hold", "x0"),
            (0.95, "CF", "hold", "x0"),
            (float("nan"), "CF", "hold", "x0"),
            (0.3, "", "hold", "x0"),
            (0.3, "CX", "hold", "x0"),
            (0.3, "F" * 17, "hold", "x0"),
            (0.3, "CSS", "hold", "x0"),
            (0.3, "SFS", "hold", "x0"),
            (0.3, "NSF", "hold", "x0"),
            (0.3, "FSN", "linear", "x0"),
            (0.3, "CF", "cubic", "x0"),
            (0.3, "CF", "hold", "eps"),
        ]
        for args in bad:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    plan.Settings.parse(*args)

    def test_characters_that_upper_turns_into_letters_are_refused(self):
        # "ﬀ".upper() is "FF" and "ß".upper() is "SS": both have to fail as foreign characters.
        for pattern in ("ﬀ", "ß", "Cﬀ", "ıF", "Ｆ"):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, "may only hold F, C, S and N"):
                    plan.Settings.parse(0.3, pattern)

    def test_surrounding_whitespace_and_lower_case_still_parse(self):
        self.assertEqual(text(plan.Settings.parse(0.3, " cCf \n").pattern), "CCF")

    def test_n_parses_alone_and_with_f_and_c(self):
        for pattern in ("N", "n", "NF", "NNF", "CNF", "nncf"):
            with self.subTest(pattern=pattern):
                self.assertEqual(text(plan.Settings.parse(0.32, pattern).pattern), pattern.upper())

    def test_n_and_s_in_one_pattern_are_refused_with_the_reason(self):
        for pattern in ("NS", "SN", "NSF", "FSN", "nsf", "CNFS"):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, "may not mix S and N"):
                    plan.Settings.parse(0.32, pattern)

    def test_only_refine_holds_an_n_step(self):
        for name, settings in plan.PRESETS.items():
            if settings is not None:
                with self.subTest(preset=name):
                    self.assertEqual(plan.Action.UNGUIDED in settings.pattern, name == "refine")


class UnguidedPlanTest(unittest.TestCase):
    """Plans of custom patterns with N steps, written out by hand: head, repeated pattern, full last step."""

    def test_pinned_plans_at_25_steps(self):
        for pattern, expected in {
            "NNF": "FFFFFFFF" + "NNFNNFNNFNNFNNFN" + "F",
            "NF": "FFFFFFFF" + "NFNFNFNFNFNFNFNF" + "F",
            "N": "FFFFFFFF" + "N" * 16 + "F",
        }.items():
            with self.subTest(pattern=pattern):
                self.assertEqual(text(plan.plan_steps(25, plan.Settings.parse(0.32, pattern))), expected)

    def test_pinned_plans_of_a_9_step_tail_cut_from_25_steps(self):
        schedule = ksampler_tail(9, 0.35)
        coverage = plan.coverage_of(schedule[0], flow_percent_to_sigma())
        for pattern, expected in {
            "NNF": "FFF" + "NNFNN" + "F",
            "NF": "FFF" + "NFNFN" + "F",
            "N": "FFF" + "NNNNN" + "F",
        }.items():
            with self.subTest(pattern=pattern):
                self.assertEqual(text(plan.plan_steps(9, plan.Settings.parse(0.32, pattern), coverage)), expected)

    def test_pinned_plan_of_7_step_refine_tail_at_denoise_035(self):
        schedule = ksampler_tail(7, 0.35)
        coverage = plan.coverage_of(schedule[0], flow_percent_to_sigma())
        p = plan.plan_steps(7, plan.PRESETS["refine"], coverage)
        self.assertEqual(text(p), "FFNNNNF")
        self.assertEqual(plan.forward_count(p), 10)

    def test_a_9_step_run_that_is_not_a_tail_stays_all_full(self):
        self.assertEqual(text(plan.plan_steps(9, plan.Settings.parse(0.32, "NNF"))), "F" * 9)

    def test_an_n_step_costs_one_forward(self):
        p = plan.plan_steps(25, plan.Settings.parse(0.32, "NNF"))
        self.assertEqual(text(p).count("F"), 14)
        self.assertEqual(text(p).count("N"), 11)
        self.assertEqual(plan.forward_count(p), 14 * 2 + 11)
        self.assertEqual(plan.forward_count(p, guided=False), 25)


if __name__ == "__main__":
    unittest.main()
