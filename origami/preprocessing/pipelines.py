from typing import Optional

from sklearn.pipeline import Pipeline

from origami.utils.config import NumericMethod, PipelineConfig, SequenceOrderMethod

from .pipes import (
    DocPermuterPipe,
    DocTokenizerPipe,
    KBinsDiscretizerPipe,
    PadTruncTokensPipe,
    SchemaParserPipe,
    TargetFieldPipe,
    TokenEncoderPipe,
    UpscalerPipe,
)


def build_prediction_pipelines(
    pipeline_config: PipelineConfig, target_field: Optional[str] = None, verbose: bool = False
) -> tuple[Pipeline, Pipeline]:
    """build common train/test pipelines for prediction tasks."""

    pipes = {
        "binning": KBinsDiscretizerPipe(
            bins=pipeline_config.n_bins, threshold=pipeline_config.n_bins, strategy="kmeans"
        ),
        "schema": SchemaParserPipe(),
        "target": TargetFieldPipe(target_field),
        "tokenizer": DocTokenizerPipe(path_in_field_tokens=pipeline_config.path_in_field_tokens),
        "padding": PadTruncTokensPipe(length="max"),
        "encoder": TokenEncoderPipe(max_tokens=pipeline_config.max_vocab_size),
    }

    # add upscaling and permuting if needed
    match pipeline_config.sequence_order:
        case SequenceOrderMethod.SHUFFLED:
            pipes |= {
                "upscaler": UpscalerPipe(n=pipeline_config.upscale),
                "permuter": DocPermuterPipe(),
            }
            train_pipes = ("schema", "upscaler", "permuter", "tokenizer", "padding", "encoder")
        case _:
            train_pipes = ("schema", "tokenizer", "padding", "encoder")

    test_pipes = ("tokenizer", "padding", "encoder")

    # add target if needed
    if target_field is not None:
        test_pipes = ("target",) + test_pipes
        train_pipes = ("target",) + train_pipes

    # add binning if needed
    if pipeline_config.numeric_method == NumericMethod.BINNING:
        test_pipes = ("binning",) + test_pipes
        train_pipes = ("binning",) + train_pipes

    train_pipeline = Pipeline([(name, pipes[name]) for name in train_pipes], verbose=verbose)
    test_pipeline = Pipeline([(name, pipes[name]) for name in test_pipes], verbose=verbose)

    if verbose:
        print(f"train pipeline: {train_pipeline}")
        print(f"test pipeline: {test_pipeline}")

    return {"train": train_pipeline, "test": test_pipeline}
