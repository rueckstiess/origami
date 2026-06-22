# Migrating from v1 to v2

Origami v2 is a breaking rewrite of the original `origami-ml` package. The package name
stays the same and imports still come from `origami`, but the public API, model internals,
CLI, configuration system, and saved checkpoint format have changed substantially.

Install v2 with:

```bash
pip install origami-ml
```

or, with uv:

```bash
uv add origami-ml
```

## What Changed

V2 is organized around a single high-level `OrigamiPipeline` API. The pipeline owns
preprocessing, tokenizer fitting, model construction, training, inference, and save/load.
Most users should start there instead of composing lower-level preprocessing/model classes
directly.

```python
from origami import OrigamiPipeline

pipeline = OrigamiPipeline()
pipeline.fit(train_data, eval_data=val_data, epochs=20)
prediction = pipeline.predict({"city": "Tokyo", "country": None}, target_key="country")
```

The command-line interface has also been rebuilt around five commands:

```bash
origami train -d train.jsonl -t label -o model.pt
origami predict -m model.pt -d test.jsonl -t label
origami generate -m model.pt -n 100
origami evaluate -m model.pt -d test.jsonl -t label
origami embed -m model.pt -d data.jsonl -o embeddings.npy
```

See [Python SDK](python-sdk.md) and [CLI Reference](cli.md) for the complete v2 surface.

## Configuration

V1 configuration was spread across CLI flags, helper modules, and model/preprocessing
objects. V2 uses nested dataclasses:

```python
from origami import DataConfig, ModelConfig, OrigamiConfig, TrainingConfig

config = OrigamiConfig(
    model=ModelConfig(d_model=256, n_layers=6),
    training=TrainingConfig(batch_size=64, num_epochs=50),
    data=DataConfig(numeric_mode="scale"),
)

pipeline = OrigamiPipeline(config)
```

Important changes:

- `OrigamiConfig` is the root config object.
- `ModelConfig` controls model architecture and position encoding.
- `TrainingConfig` controls optimization, evaluation, callbacks, and training constraints.
- `DataConfig` controls numeric preprocessing, schema inference, and vocabulary pruning.
- `InferenceConfig` controls inference-time grammar/schema constraints separately from training.

See [Configuration](configuration.md) for all fields.

## Removed or Reworked APIs

The v2 codebase removes or replaces several v1-era modules. In particular:

- `origami.preprocessing.pipes`, `pipelines`, `encoder`, and dataframe dataset helpers are gone.
- v1 Monte Carlo and rejection estimator modules are not part of the v2 package surface.
- v1 sampler/autocomplete internals are replaced by `origami.inference.OrigamiGenerator`.
- model internals moved from `origami.model.origami`, `positions`, and `vpda` into the new
  tokenizer, constraints, position encoding, model, training, pipeline, and inference packages.

For field prediction, use:

```python
pipeline.predict(obj, target_key="label")
pipeline.predict_batch(objects, target_key="label")
pipeline.predict_proba(obj, target_key="label", top_k=5)
```

For lower-level generation, use `OrigamiGenerator`; predictors delegate to the generator rather
than implementing their own decoding logic.

## Numeric Fields

V2 has explicit numeric handling modes:

- `numeric_mode="disabled"`: treat all numbers as discrete tokens.
- `numeric_mode="discretize"`: bin high-cardinality numeric fields.
- `numeric_mode="scale"`: normalize high-cardinality numerics and use the continuous output head.

```python
from origami import DataConfig, OrigamiConfig

config = OrigamiConfig(data=DataConfig(numeric_mode="scale", cat_threshold=100))
```

See [Concepts: Handling Numeric Fields](concepts.md#handling-numeric-fields) for guidance.

## Checkpoints

V1 checkpoints are not expected to load in v2. Train new v2 models and save them with:

```python
pipeline.save("model.pt")
loaded = OrigamiPipeline.load("model.pt")
```

V2 checkpoints include the pipeline state needed for inference: model, tokenizer,
configuration, preprocessing state, and schema state where applicable.

## Practical Migration Path

1. Start with `OrigamiPipeline` and a small representative dataset.
2. Recreate your old training setup with `OrigamiConfig`, `ModelConfig`, `TrainingConfig`,
   and `DataConfig`.
3. Replace direct sampler/predictor internals with `pipeline.predict`, `pipeline.generate`,
   `pipeline.evaluate`, and `pipeline.embed`.
4. Retrain and save new v2 checkpoints.
5. If you depended on v1 estimator or experiment modules, keep using v1 for that workflow or
   port the workflow explicitly onto v2 inference primitives.
