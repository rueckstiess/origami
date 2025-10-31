# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-10-31

### Added
- New `origami embed` CLI command for generating embeddings from trained models
- Progress bars with proper updates for embedding generation

### Fixed
- Fixed device mismatch bug in `Embedder` class when using mean pooling with MPS/CUDA devices
- Fixed progress bar not updating during embedding generation in `origami embed` command

### Changed
- Updated minimum Python version to 3.11

### Removed
- Removed guildai dependency from main codebase

## [0.2.0] - 2025-10-23

### Added
- `Sampler` class for generating unbiased samples from learned model distribution
- `SortFieldsPipe` preprocessing pipe to ensure consistent field ordering in documents
- CI test infrastructure
- `MCEstimator` and `RejectionEstimator` for query selectivity estimation
- Support for MongoDB +srv URIs

### Changed
- Migrated build system to `uv` from traditional pip/setuptools workflow
- Code formatting with ruff
- Updated README documentation

### Fixed
- Fixed nested tensor warning
- Fixed duplicate sampling bug in `Sampler` class
- Skip bad samples in `Sampler` to avoid 'pipe not fitted' warnings

### Removed
- Removed `setup.cfg` in favor of `pyproject.toml` configuration

[0.3.0]: https://github.com/rueckstiess/origami/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rueckstiess/origami/releases/tag/v0.2.0
