import pathlib
import warnings

import click
from click_option_group import optgroup
from omegaconf import OmegaConf

from storm_ml.inference import Predictor
from storm_ml.model import STORM
from storm_ml.model.vpda import DocumentVPDA
from storm_ml.preprocessing import (
    DFDataset,
    build_prediction_pipelines,
)
from storm_ml.utils import TopLevelConfig, save_storm_model

from .utils import create_projection, load_data, make_progress_callback

# suppress deprecation warning for setting epoch in LR scheduler, likely bug in pytorch
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.optim.lr_scheduler",
    message=r".*The epoch parameter in `scheduler\.step\(\)`.*",
)


@click.command()
@click.argument("source", type=str)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(dir_okay=False, path_type=pathlib.Path, resolve_path=True),
    default="./model.storm",
    show_default=True,
    help="path to write trained model",
)
@optgroup.group("Source Options")
@optgroup.option("--source-db", "-d", type=str, help="database name, only used when SOURCE is a MongoDB URI.")
@optgroup.option("--source-coll", "-c", type=str, help="collection name, only used when SOURCE is a MongoDB URI.")
@optgroup.option("--include-fields", "-i", type=str, help="comma-separated list of field names to include")
@optgroup.option("--exclude-fields", "-e", type=str, help="comma-separated list of field names to exclude")
@optgroup.option("--limit", "-l", type=int, default=0, help="limit the number of documents to load")
@optgroup.group("Config Options")
@optgroup.option("--config-file", "-C", type=click.File("r"), help="path to config file")
@optgroup.option(
    "--max-vocab-size",
    "-V",
    type=int,
    default=0,
    show_default=True,
    help="maximum number of tokens in the vocabulary",
)
@optgroup.option(
    "--num-layers",
    "-L",
    type=int,
    default=3,
    show_default=True,
    help="number of transformer layers",
)
@optgroup.option(
    "--num-attn-heads",
    "-A",
    type=int,
    default=4,
    show_default=True,
    help="number of attention heads",
)
@optgroup.option(
    "--hidden-dim",
    "-H",
    type=int,
    default=128,
    show_default=True,
    help="hidden dimensionality of transformer layers",
)
@optgroup.option(
    "--num-batches", "-N", type=int, default=10000, show_default=True, help="number of batches to train on"
)
@optgroup.option("--batch-size", "-B", type=int, default=100, show_default=True, help="batch size")
@optgroup.option(
    "--pos-encoding",
    "-P",
    type=click.Choice(["NONE", "INTEGER", "KEY_VALUE"]),
    help="type of position encoding",
    default="KEY_VALUE",
    show_default=True,
)
@optgroup.option(
    "--upscaling",
    "-U",
    type=int,
    help="upscaling factor, when `--shuffled` mode is used",
    default=5,
    show_default=True,
)
@optgroup.option(
    "--shuffled/--ordered",
    "shuffled",
    is_flag=True,
    default=True,
    help="shuffle key/value pairs in each object",
    show_default=True,
)
@optgroup.group("Evaluation Options")
@optgroup.option(
    "--eval-mode",
    type=click.Choice(["none", "test-split", "cross-val", "eval-source"]),
    help="evaluation mode",
    default="none",
    show_default=True,
)
@optgroup.option("--split-ratio", type=float, default=0.2, show_default=True, help="test split ratio")
@optgroup.option("--k-fold", type=int, default=5, show_default=True, help="number of folds for cross-validation")
@optgroup.option("--eval-source", type=click.Path(exists=True), help="path to evaluation source")
@optgroup.option("--target-field", "-t", type=str, help="target field name to predict")
@click.option("--verbose", "-v", is_flag=True, default=True)
def train(source: str, **kwargs):
    """
    Train a STORM model.
    """
    config = TopLevelConfig()

    # data configs
    config.data.source = source
    config.data.db = kwargs["source_db"]
    config.data.coll = kwargs["source_coll"]
    config.data.projection = create_projection(kwargs["include_fields"], kwargs["exclude_fields"])
    config.data.limit = kwargs["limit"]
    config.data.target_field = kwargs["target_field"]

    # model configs
    config.model.n_layer = kwargs["num_layers"]
    config.model.n_head = kwargs["num_attn_heads"]
    config.model.n_embd = kwargs["hidden_dim"]
    config.model.position_encoding = kwargs["pos_encoding"]

    # train configs
    config.train.n_batches = kwargs["num_batches"]
    config.train.batch_size = kwargs["batch_size"]
    config.train.learning_rate = 1e-3
    config.train.n_warmup_batches = 1000
    config.train.print_every = 10
    config.train.eval_every = 100

    # pipeline configs
    config.pipeline.max_vocab_size = kwargs["max_vocab_size"]
    config.pipeline.test_split = kwargs["split_ratio"]
    config.pipeline.sequence_order = "SHUFFLED" if kwargs["shuffled"] else "ORDERED"
    config.pipeline.upscale = kwargs["upscaling"]

    # load data
    df = load_data(source, config.data)

    # build pipelines
    pipelines = build_prediction_pipelines(config.pipeline, config.data.target_field)

    # process train, eval and test data
    train_df = pipelines["train"].fit_transform(df)
    # test_df = pipeline.transform(test_docs_df)

    # get stateful objects and set model parameters
    schema = pipelines["train"]["schema"].schema
    encoder = pipelines["train"]["encoder"].encoder
    config.model.block_size = pipelines["train"]["padding"].length
    config.model.vocab_size = encoder.vocab_size

    # datasets
    train_dataset = DFDataset(train_df)

    vpda = DocumentVPDA(encoder, schema)
    model = STORM(config.model, config.train, vpda=vpda)

    if kwargs["verbose"]:
        print(OmegaConf.to_yaml(config))

    # model callback during training, prints training and test metrics
    if config.data.target_field:
        predictor = Predictor(model, encoder, config.data.target_field, max_batch_size=config.train.batch_size)
    else:
        predictor = None
    progress_callback = make_progress_callback(config.train, train_dataset, predictor=predictor)

    # train model
    model.set_callback("on_batch_end", progress_callback)
    model.train_model(train_dataset, batches=config.train.n_batches)

    # save model with config
    save_storm_model(model, pipelines=pipelines, config=config, path=kwargs.get("model_path"))
