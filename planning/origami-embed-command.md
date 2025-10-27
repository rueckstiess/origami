# Implementation Plan: `origami embed` Command

**Date:** 2025-10-27
**Status:** Planning
**Assignee:** Claude Code

---

## Overview

Add a new CLI command `origami embed` to generate embeddings from trained ORIGAMI models. The command will follow the structure of the existing `predict` command and support multiple output formats with intelligent format detection from file extensions.

## Objectives

- Provide a simple CLI interface for generating embeddings from trained models
- Support multiple output formats (CSV, JSON, JSONL, Torch, Numpy)
- Automatically detect output format from file extension
- Support optional L2 normalization for cosine similarity use cases
- Maintain consistency with existing CLI commands (train, predict)

---

## Files to Create/Modify

### 1. **Create** `origami/cli/embed.py` (new file)

**Responsibilities:**
- Define Click command with appropriate options and option groups
- Load model, config, and pipelines using `load_origami_model()`
- Initialize ORIGAMI model with VPDA guardrails
- Load and transform source data
- Create Embedder instance and generate embeddings
- Apply optional L2 normalization
- Save embeddings in detected format

**Key Components:**
- Command decorator with argument/options
- Model loading logic (similar to predict.py:42-46)
- Data loading and transformation (similar to predict.py:64-82)
- Embedder initialization and execution
- Progress bar using tqdm
- Normalization logic
- Output format detection and saving
- Verbose output handling

### 2. **Modify** `origami/cli/main.py`

**Changes:**
```python
from origami.cli.embed import embed
main.add_command(embed)
```

### 3. **Extend** `origami/cli/utils.py`

**New Functions:**

1. `infer_output_format(file_path: pathlib.Path) -> str`
   - Detect format from file extension
   - Supported extensions: `.csv`, `.json`, `.jsonl`, `.pt`, `.pth`, `.npy`
   - Raise error for unsupported extensions

2. `save_embeddings(embeddings: np.ndarray, output_file: pathlib.Path, format: str) -> None`
   - Handle all supported output formats
   - Convert tensor to appropriate format for each output type

---

## CLI Interface

### Command Signature
```bash
origami embed [OPTIONS] SOURCE
```

### Arguments
- `SOURCE` (required): MongoDB connection string or file path (csv/json/jsonl)

### Options

**Model Options:**
- `-m, --model-path FILE` (required): Path to trained `.origami` model

**Source Options:**
- `-d, --source-db TEXT`: Database name (required for MongoDB URI)
- `-c, --source-coll TEXT`: Collection name (required for MongoDB URI)
- `-i, --include-fields TEXT`: Comma-separated list of fields to include
- `-e, --exclude-fields TEXT`: Comma-separated list of fields to exclude
- `-s, --skip INTEGER`: Number of documents to skip (default: 0)
- `-l, --limit INTEGER`: Limit number of documents to load (default: 0, no limit)

**Embedding Options:**
- `-p, --position {target|last|end}`: Position to extract embedding from (default: "last")
  - `target`: Extract embedding at target field position (requires model trained with --target-field)
  - `last`: Extract embedding from last non-padding token
  - `end`: Extract embedding from sequence end token
- `-r, --reduction {index|sum|mean}`: Reduction/pooling strategy (default: "index")
  - `index`: Extract embedding at specific position
  - `sum`: Sum embeddings across sequence
  - `mean`: Average embeddings across sequence
- `--normalize`: L2-normalize embeddings (flag, default: False)

**Output Options:**
- `-o, --output-file PATH` (required): File to store embeddings
  - Format automatically detected from extension

**General Options:**
- `-v, --verbose`: Enable verbose output (flag)
- `--help`: Show help message

### Usage Examples

```bash
# Generate embeddings from MongoDB, save as numpy
origami embed mongodb://localhost:27017 \
  -d mydb -c mycoll \
  -m model.origami \
  -o embeddings.npy

# Generate normalized embeddings for cosine similarity
origami embed data.json \
  -m model.origami \
  -o embeddings.csv \
  --normalize

# Extract embeddings at target position with mean pooling
origami embed data.jsonl \
  -m model.origami \
  -p target -r mean \
  -o embeddings.pt

# Generate embeddings with field filtering
origami embed mongodb://localhost:27017 \
  -d mydb -c mycoll \
  -m model.origami \
  -e "_id,metadata" \
  -l 1000 \
  -o embeddings.json
```

---

## Implementation Details

### Main Logic Flow

```python
def embed(source: str, **kwargs):
    # 1. Load model and pipelines
    model, pipelines, config = load_origami_model(kwargs["model_path"])

    # 2. Validate position="target" requires target_field
    if kwargs["position"] == "target" and config.data.target_field is None:
        raise click.BadParameter(
            "--position=target requires a target field. "
            "Model was trained without --target-field."
        )

    # 3. Initialize model with VPDA (from saved config)
    # Similar to predict.py:50-61

    # 4. Load and transform data
    df = load_data(source, config.data)
    test_df = pipelines["test"].transform(df)
    dataset = DFDataset(test_df)

    # 5. Initialize embedder
    embedder = Embedder(
        model,
        encoder,
        config.data.target_field,
        batch_size=config.train.batch_size
    )

    # 6. Generate embeddings with progress bar
    with tqdm(total=len(dataset), disable=not kwargs["verbose"]) as pbar:
        embeddings = embedder.embed(
            dataset,
            kwargs["position"],
            kwargs["reduction"]
        )

    # 7. Move to CPU
    embeddings = embeddings.cpu()

    # 8. Optional L2 normalization
    if kwargs["normalize"]:
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        embeddings = embeddings / norms.clamp(min=1e-12)

    # 9. Infer format and save
    output_path = pathlib.Path(kwargs["output_file"])
    format = infer_output_format(output_path)
    save_embeddings(embeddings.numpy(), output_path, format)

    # 10. Verbose output
    if kwargs["verbose"]:
        click.echo(f"Generated embeddings: shape={embeddings.shape}")
        click.echo(f"Position={kwargs['position']}, Reduction={kwargs['reduction']}")
        click.echo(f"Normalized={kwargs['normalize']}")
        click.echo(f"Saved to: {output_path} (format: {format})")
```

### Output Format Handlers

#### Format Detection
```python
def infer_output_format(file_path: pathlib.Path) -> str:
    """Infer output format from file extension."""
    suffix = file_path.suffix.lower()
    format_map = {
        '.csv': 'csv',
        '.json': 'json',
        '.jsonl': 'jsonl',
        '.pt': 'torch',
        '.pth': 'torch',
        '.npy': 'numpy',
    }
    if suffix not in format_map:
        raise ValueError(
            f"Unsupported output format: {suffix}. "
            f"Supported: {', '.join(format_map.keys())}"
        )
    return format_map[suffix]
```

#### CSV Format (`.csv`)
```python
# Convert to DataFrame with columns: emb_0, emb_1, ..., emb_n
import pandas as pd
df = pd.DataFrame(
    embeddings,
    columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
)
df.to_csv(output_file, index=False)
```

#### JSON Format (`.json`)
```python
# Save as nested array
import json
with open(output_file, 'w') as f:
    json.dump(embeddings.tolist(), f, indent=2)
```

#### JSONL Format (`.jsonl`)
```python
# One embedding array per line
import json
with open(output_file, 'w') as f:
    for row in embeddings:
        f.write(json.dumps(row.tolist()) + '\n')
```

#### Torch Format (`.pt`, `.pth`)
```python
# Save as PyTorch tensor (already on CPU)
import torch
torch.save(torch.tensor(embeddings), output_file)
```

#### Numpy Format (`.npy`)
```python
# Save as NumPy array
import numpy as np
np.save(output_file, embeddings)
```

### L2 Normalization

**Purpose:** Normalize embeddings to unit length for cosine similarity comparisons.

**Mathematical Definition:**
- For embedding vector `v`: `v_normalized = v / ||v||_2`
- After normalization: `||v_normalized||_2 = 1.0`
- Cosine similarity becomes dot product: `cos(A, B) = A · B` (when both normalized)

**Implementation:**
```python
if normalize:
    # L2 normalize each embedding to unit length
    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    embeddings = embeddings / norms.clamp(min=1e-12)  # avoid division by zero
```

**Benefits:**
- Faster similarity computation (dot product vs full cosine formula)
- Common in nearest neighbor search and clustering
- Standardizes embedding magnitudes

---

## Validation & Error Handling

### Position Validation
```python
if position == "target" and config.data.target_field is None:
    raise click.BadParameter(
        "--position=target requires a target field. "
        "Model was trained without --target-field."
    )
```

### Format Validation
```python
# In infer_output_format()
if suffix not in format_map:
    raise ValueError(
        f"Unsupported output format: {suffix}. "
        f"Supported: .csv, .json, .jsonl, .pt, .pth, .npy"
    )
```

### MongoDB Validation
```python
if source.startswith("mongodb://"):
    if kwargs["source_db"] is None or kwargs["source_coll"] is None:
        raise click.BadParameter(
            "--source-db and --source-coll are required for MongoDB URI"
        )
```

### Output File Validation
```python
output_path = pathlib.Path(kwargs["output_file"])
if not output_path.parent.exists():
    raise click.BadParameter(
        f"Output directory does not exist: {output_path.parent}"
    )
```

### Empty Dataset
```python
if len(dataset) == 0:
    raise click.ClickException("No documents loaded from source")
```

---

## Design Decisions & Rationale

### 1. No Document Metadata (IDs/Indices)
**Decision:** Do not include document IDs or row indices in output.

**Rationale:**
- Embeddings maintain source order automatically
- Users can align embeddings with source data by index
- Keeps output format clean and focused on embeddings
- Can be added later if needed without breaking changes

### 2. No Batch Size Option
**Decision:** Use `config.train.batch_size` from saved model.

**Rationale:**
- Reduces CLI complexity
- Batch size doesn't significantly affect output (unlike training)
- Can be added later if users request customization
- Keeps interface consistent with predict command

### 3. Format Detection from Extension
**Decision:** Infer format from file extension, no explicit `--output-format` flag.

**Rationale:**
- More intuitive user experience
- Reduces CLI clutter
- File extension already indicates format
- Prevents mismatches between extension and format flag
- Standard practice in many tools

### 4. Target Field from Saved Config
**Decision:** Use `config.data.target_field` from model, don't allow override.

**Rationale:**
- Target field pipeline is built into saved model
- Consistent with how model was trained
- Prevents confusion and errors
- Users can train separate models for different target fields

### 5. Always Run on CPU
**Decision:** Move embeddings to CPU before saving, regardless of training device.

**Rationale:**
- Output file portability (can load on systems without GPU)
- Inference on CPU is often faster on laptops
- User can test performance and configure later if needed
- Simplifies output handling (no device checks)

---

## Testing Checklist

After implementation, verify:

### Functionality
- [ ] Load model from `.origami` file successfully
- [ ] Generate embeddings with `position="target"` (requires target field)
- [ ] Generate embeddings with `position="last"`
- [ ] Generate embeddings with `position="end"`
- [ ] Generate embeddings with `reduction="index"`
- [ ] Generate embeddings with `reduction="sum"`
- [ ] Generate embeddings with `reduction="mean"`
- [ ] Apply `--normalize` flag and verify norm ≈ 1.0

### Output Formats
- [ ] Save as CSV (`.csv`) and verify format
- [ ] Save as JSON (`.json`) and verify structure
- [ ] Save as JSONL (`.jsonl`) and verify one array per line
- [ ] Save as Torch (`.pt`, `.pth`) and verify loadable with `torch.load()`
- [ ] Save as Numpy (`.npy`) and verify loadable with `np.load()`

### Data Sources
- [ ] Load from MongoDB with connection string
- [ ] Load from JSON file (`.json`)
- [ ] Load from JSONL file (`.jsonl`)
- [ ] Load from CSV file (`.csv`)

### Options & Filters
- [ ] Apply `--include-fields` filter
- [ ] Apply `--exclude-fields` filter
- [ ] Apply `--skip` option
- [ ] Apply `--limit` option
- [ ] Verbose mode shows expected output

### Error Handling
- [ ] Error when `position="target"` but model has no target field
- [ ] Error for unsupported file extension
- [ ] Error when MongoDB URI missing `--source-db` or `--source-coll`
- [ ] Error when output directory doesn't exist
- [ ] Error when source has no documents

### Edge Cases
- [ ] Empty dataset (0 documents after filters)
- [ ] Very large dataset (progress bar works correctly)
- [ ] Unicode characters in field names
- [ ] Embeddings maintain source document order

---

## Future Enhancements (Out of Scope)

These features are explicitly **not** included in the initial implementation but could be added later:

1. **Document Metadata Output**
   - `--include-index` flag to add row indices to CSV output
   - `--include-id` to preserve document IDs

2. **Batch Size Configuration**
   - `--batch-size` option to override saved config

3. **Additional Output Formats**
   - Parquet (`.parquet`) for large embeddings
   - HDF5 (`.h5`) for scientific workflows
   - Arrow (`.arrow`) for cross-language compatibility

4. **Precision Control**
   - `--precision {float16|float32|float64}` for file size control

5. **Metadata Sidecar**
   - Companion JSON file with embedding metadata (position, reduction, model path, timestamp)

6. **Compression**
   - `--compress` flag for gzip compression of output files

7. **Streaming Mode**
   - Process and save embeddings in chunks for very large datasets

---

## Questions & Clarifications

### Resolved
- ✅ Document metadata → Not needed (maintain source order)
- ✅ Batch size option → Use saved config
- ✅ Output format → Infer from extension
- ✅ Target field pipeline → Loaded from model
- ✅ L2 normalization → Normalize to unit length (norm = 1.0)

### Open Questions
None currently.

---

## Implementation Timeline

1. **Phase 1:** Create `origami/cli/embed.py` with core functionality
2. **Phase 2:** Add utility functions to `origami/cli/utils.py`
3. **Phase 3:** Register command in `origami/cli/main.py`
4. **Phase 4:** Test all output formats and data sources
5. **Phase 5:** Validate error handling and edge cases
6. **Phase 6:** Update documentation (CLI.md)

---

## References

- Existing implementation: [origami/cli/predict.py](origami/cli/predict.py)
- Embedder class: [origami/inference/embedder.py](origami/inference/embedder.py)
- Utilities: [origami/cli/utils.py](origami/cli/utils.py)
- Model loader: [origami/utils/common.py:335](origami/utils/common.py#L335)
