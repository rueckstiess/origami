# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Loss is now an opt-in metric in `evaluate()` / `pipeline.evaluate()`.** Loss
  is requested with the reserved metric spec `"loss"`, like any other metric:
  ```python
  pipeline.evaluate(data)                                              # loss only (default)
  pipeline.evaluate(data, metrics={"acc": "accuracy"})                 # accuracy only, no loss
  pipeline.evaluate(data, metrics={"loss": "loss", "acc": "accuracy"}) # both
  ```
  When no `metrics` are passed, loss is still computed by default. This avoids
  paying for the loss pass when only prediction metrics are needed.

  **Breaking:** passing a `metrics` dict that does not include `"loss"` no longer
  returns a `"loss"` key in the results. Training behavior is unchanged — the
  trainer always computes loss for `best_metric` tracking, checkpointing, and
  progress display.
