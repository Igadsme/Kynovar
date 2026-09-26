# Milestone 7 — Theory Revision

Status: implemented and measured. Source: `results/theory_shift/m7/theory_shift.json`.

## Setting

A `ChangingLaboratory` (evaluation-only, `kynovar/evaluation/worlds.py`) silently switches the hidden force from `2.5 m1 m2 / r^3` to `2.5 m1 m2 / r^1.5` after experiment 24. The scientist is not told that a change exists. Instrument noise is 3% relative + 0.005 absolute. Budget: 50 experiments per stream.

## Method (`kynovar/theory/revision.py`)

1. **Initial discovery** on 6 experiments. The resulting law becomes `Law K-17 v1`, and its fitted noise level becomes the reference noise.
2. **Residual monitor.** For every new experiment, compute `log mean z^2` of the observations under the current law. A one-sided CUSUM with k = 0.5 and h = 5 runs over this statistic after 6 warm-up experiments.
3. **Investigation** after an alarm. Run 6 new experiments and rediscover a law using *only* post-alarm experiments. The change point is estimated as the likelihood argmax over candidate split points.
4. **Validation** on 3 further experiments. The candidate is adopted only if it supports all 3, the old law contradicts at least one, and the candidate is *sharp*: fitted relative noise at most 2× the reference, and absolute noise at most 2× the reference.
5. **Versioning.** An adopted candidate becomes `Law K-17 v2` with `valid_from` set to the estimated change point; v1 gets `valid_to`. Each version records its evidence experiments, reason, and parameter differences.
6. After a failed validation only the CUSUM restarts; the v1 baseline is kept.

## Acceptance experiment

20 change streams (seeds 0–19) and 20 control streams without a change (seeds 1000–1019).

```bash
.venv/bin/python scripts/theory_shift.py --seeds 20
```

| Metric | Value |
| --- | ---: |
| v1 recovered the pre-change law | 40 / 40 streams |
| Change detected | 20 / 20 |
| Detection delay (experiments after the change) | mean 2.25, std 2.38, median 1, n = 20 |
| Replacement validated and adopted | 19 / 20 |
| Final law recovers the post-change law | 19 / 20 |
| Change-point estimate error (adopted streams) | 0 experiments in 19 / 19 |
| Control streams with any false alarm | 1 / 20 |
| Control false alarms per monitored experiment | 1 / 760 (0.13%) |
| Pre-change false alarms in change streams | 5 in 240 monitored experiments (2.1%) |
| Runtime | 153 s |

Pooled over control streams and pre-change segments, the false-alarm rate is 6 / 1000 ≈ 0.6% per monitored experiment.

No false alarm led to adopting a wrong law: every adopted v2 in the change streams has `valid_from = 24` and the post-change structure, and the single control alarm (seed 1019, experiment 16) failed validation, so v1 stayed in place. The sharpness requirement exists because an earlier version did so: a pre-change false alarm plus a mixed investigation window produced an over-dispersed "law" that passed the support test.

## Failure: seed 9

Seed 9 detected the change after 1 experiment, and all three investigations found the correct law (`2.52 m1 m2 / r^1.508`, `2.50 / r^1.502`, `2.50 / r^1.504`). Each candidate supported all 3 validation experiments while the old law was contradicted. Every candidate was rejected by the sharpness test. The reference absolute noise fitted on the first 6 experiments was unusually small: a = 0.0034, the smallest of the 32 reference values across all investigations (median 0.0087, max 0.0131). The candidates' absolute-noise terms (0.008–0.022) exceeded 2× that value, while their relative terms were below the reference.

The maximum-likelihood fit can trade noise between the absolute and the relative component, so a component-wise comparison is brittle. A likely fix is to compare the total predicted σ on the investigation data. That fix was designed after seeing this failure. It has not been evaluated, and it must be tested on fresh seeds rather than on these 20. The 19/20 result stands as measured.

## Limitations

- One kind of change (exponent jump from 3 to 1.5) at one noise level. Smaller changes will take longer to detect; that was not measured.
- The CUSUM parameters (k = 0.5, h = 5) were set by hand before the acceptance run.
- The change-point estimate is exact here because the jump is large; it is not evidence of precision on subtle changes.
- Only one change per stream.
