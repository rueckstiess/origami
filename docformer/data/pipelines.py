from sklearn.pipeline import Pipeline

from docformer.utils import SequenceOrderMethod

from .pipes import (
    DocPermuterPipe,
    DocTokenizerPipe,
    ExistsTrackerPipe,
    IdSetterPipe,
    KBinsDiscretizerPipe,
    PadTruncTokensPipe,
    SchemaParserPipe,
    TargetFieldPipe,
    TokenEncoderPipe,
    UpscalerPipe,
)


def build_prediction_pipelines(
    target_field: str, n_bins: int, sequence_order: SequenceOrderMethod, upscale: int
) -> tuple[Pipeline, Pipeline]:
    """build common train/test pipelines for prediction tasks."""

    pipes = {
        "binning": KBinsDiscretizerPipe(bins=n_bins, threshold=n_bins, strategy="kmeans"),
        "schema": SchemaParserPipe(),
        "target": TargetFieldPipe(target_field),
        "tokenizer": DocTokenizerPipe(),
        "padding": PadTruncTokensPipe(length="max"),
        "encoder": TokenEncoderPipe(),
    }
    match sequence_order:
        case SequenceOrderMethod.SHUFFLED:
            pipes |= {
                "upscaler": UpscalerPipe(n=upscale),
                "permuter": DocPermuterPipe(),
            }
            train_pipes = ("binning", "schema", "upscaler", "permuter", "tokenizer", "padding", "encoder")
        case _:
            train_pipes = ("binning", "schema", "tokenizer", "padding", "encoder")

    test_pipes = ("binning", "target", "tokenizer", "padding", "encoder")

    train_pipeline = Pipeline([(name, pipes[name]) for name in train_pipes])
    test_pipeline = Pipeline([(name, pipes[name]) for name in test_pipes])

    return train_pipeline, test_pipeline


def build_estimation_pipeline(n_bins: int, sequence_order: SequenceOrderMethod, keep_id: bool = False) -> Pipeline:
    """build common pipeline for estimation tasks."""

    match sequence_order:
        case SequenceOrderMethod.SHUFFLED:
            permuter = [("permuter", DocPermuterPipe())]
        case _:
            permuter = []

    if keep_id:
        id_setter = [("id_setter", IdSetterPipe())]
    else:
        id_setter = []

    pipeline = Pipeline(
        [
            ("binning", KBinsDiscretizerPipe(bins=n_bins)),
            # only include IdSetterPipe if keep_id is True
            *id_setter,
            ("schema", SchemaParserPipe()),
            ("exists", ExistsTrackerPipe()),
            *permuter,
            ("tokenizer", DocTokenizerPipe()),
            ("padding", PadTruncTokensPipe(length="max")),
            ("encoder", TokenEncoderPipe()),
        ]
    )

    return pipeline
