# rreal test suite

Pytest-based tests for rreal, following the same conventions as the adarl suite. The suite is
always runnable: anything needing a GPU skips itself when the requirement is missing.

## Layout

```
tests/
  conftest.py                                 # layout docs, per-test RNG seeding
  feature_extractors/
    test_mixed_feature_extractor.py           # MixedFeatureExtractor: vector+image dict observations
```

## Running

The project venv is `…/virtualenv/host313`. `pytest` is in `[project.optional-dependencies].test`.

```bash
cd <repo root>                          # dir with pyproject.toml
python -m pytest                        # everything
python -m pytest -m "not gpu"           # fast CPU lane
python -m pytest tests/feature_extractors
```

Markers: `gpu`, `slow`, `integration` (`--strict-markers` is on).

## Roadmap — tests worth adding

- `nets/`: `Mixed_encoder` / `Image_encoder` shape contracts across backbones (`conv`,
  `mobilenetv3`, `resnet18`, `bigconv`), ensemble sizes > 1, `combiner_arch="identity"`.
- `StackVectorsFeatureExtractor`: same coverage as the mixed one. Note its `load()` currently
  falls off the end and returns `None`, so a round-trip test would fail as written.
- `build_mlp_net` / `Parallel`: ensemble mean/std reduction, weightnorm, init multipliers.
- A short training smoke (a few SAC/PPO iterations) with a mixed observation space, to catch
  breakage where the feature extractor meets the algorithms.
