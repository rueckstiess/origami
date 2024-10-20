import pathlib
import warnings

import click
from click_option_group import optgroup
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

from storm_ml.inference import Predictor
from storm_ml.model import STORM
from storm_ml.model.vpda import DocumentVPDA
from storm_ml.preprocessing import (
    DFDataset,
    DocTokenizerPipe,
    PadTruncTokensPipe,
    SchemaParserPipe,
    TokenEncoderPipe,
)
from storm_ml.utils import TopLevelConfig, save_model
from storm_ml.utils.guild import print_guild_scalars

from .utils import create_projection, load_data

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
    default="./storm.pt",
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
    default=512,
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
@click.option("--verbose", "-v", is_flag=True)
def train(source: str, **kwargs):
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
    config.train.learning_rate = 1e-3
    config.train.n_warmup_batches = 1000

    # pipeline configs
    config.pipeline.max_vocab_size = kwargs["max_vocab_size"]
    config.pipeline.test_split = kwargs["split_ratio"]
    config.pipeline.sequence_order = "SHUFFLED" if kwargs["shuffled"] else "ORDERED"
    config.pipeline.upscale = kwargs["upscaling"]

    # load data
    df = load_data(source, **kwargs)

    # build pipelines
    pipes = {
        "schema": SchemaParserPipe(),
        "tokenizer": DocTokenizerPipe(path_in_field_tokens=True),
        "padding": PadTruncTokensPipe(length="max"),
        "encoder": TokenEncoderPipe(max_tokens=config.pipeline.max_vocab_size),
    }
    train_pipeline = Pipeline([(name, pipes[name]) for name in ("schema", "tokenizer", "padding", "encoder")])

    # process train, eval and test data
    train_df = train_pipeline.fit_transform(df)
    # test_df = pipeline.transform(test_docs_df)

    # get stateful objects and set model parameters
    schema = pipes["schema"].schema
    encoder = pipes["encoder"].encoder
    config.model.block_size = pipes["padding"].length
    config.model.vocab_size = encoder.vocab_size

    # datasets
    train_dataset = DFDataset(train_df)

    vpda = DocumentVPDA(encoder, schema)
    model = STORM(config.model, config.train, vpda=vpda)

    if kwargs["verbose"]:
        print(OmegaConf.to_yaml(config))

    # model callback during training, prints training and test metrics
    def progress_callback(model):
        if model.batch_num % config.train.eval_every == 0:
            print_guild_scalars(
                step=f"{int(model.batch_num / config.train.eval_every)}",
                epoch=model.epoch_num,
                batch_num=model.batch_num,
                batch_dt=f"{model.batch_dt*1000:.2f}",
                batch_loss=f"{model.loss:.4f}",
                # test_loss=f"{predictor.ce_loss(test_dataset.sample(n=100)):.4f}",
                # test_acc=f"{predictor.accuracy(test_dataset.sample(n=100)):.4f}",
                lr=f"{model.learning_rate:.2e}",
            )

    # train model
    model.set_callback("on_batch_end", progress_callback)
    model.train_model(train_dataset, batches=config.train.n_batches)

    # save model with config
    save_model(model, config, kwargs.get("model_path"))
