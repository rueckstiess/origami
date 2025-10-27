import pathlib

import click
import torch
from click_option_group import optgroup
from omegaconf import OmegaConf
from tqdm import tqdm

from origami.cli.utils import create_projection, infer_output_format, load_data, save_embeddings
from origami.inference import Embedder
from origami.model import ORIGAMI
from origami.model.vpda import ObjectVPDA
from origami.preprocessing import DFDataset
from origami.utils import count_parameters, load_origami_model
from origami.utils.config import GuardrailsMethod


@click.command()
@click.argument("source", type=str)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path, resolve_path=True),
    required=True,
    help="path to trained model",
)
@optgroup.group("Source Options")
@optgroup.option("--source-db", "-d", type=str, help="database name, only used when SOURCE is a MongoDB URI.")
@optgroup.option("--source-coll", "-c", type=str, help="collection name, only used when SOURCE is a MongoDB URI.")
@optgroup.option("--include-fields", "-i", type=str, help="comma-separated list of field names to include")
@optgroup.option("--exclude-fields", "-e", type=str, help="comma-separated list of field names to exclude")
@optgroup.option("--skip", "-s", type=int, default=0, help="number of documents to skip")
@optgroup.option("--limit", "-l", type=int, default=0, help="limit the number of documents to load")
@optgroup.group("Embedding Options")
@optgroup.option(
    "--position",
    "-p",
    type=click.Choice(["target", "last", "end"]),
    default="last",
    show_default=True,
    help="position to extract embedding from",
)
@optgroup.option(
    "--reduction",
    "-r",
    type=click.Choice(["index", "sum", "mean"]),
    default="index",
    show_default=True,
    help="reduction/pooling strategy",
)
@optgroup.option(
    "--normalize",
    is_flag=True,
    default=False,
    help="L2-normalize embeddings to unit length",
)
@optgroup.group("Output Options")
@optgroup.option(
    "--output-file",
    "-o",
    type=click.Path(path_type=pathlib.Path, resolve_path=True),
    required=True,
    help="file to store embeddings (format inferred from extension)",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def embed(source, **kwargs):
    """Generate embeddings with a trained ORIGAMI model."""

    # load model, config and pipelines
    model_dict = load_origami_model(kwargs.get("model_path"))
    state_dict = model_dict["state_dict"]
    config = model_dict["config"]
    pipelines = model_dict["pipelines"]
    encoder = pipelines["train"]["encoder"].encoder
    schema = pipelines["train"]["schema"].schema

    # validate position="target" requires target_field
    if kwargs["position"] == "target" and config.data.target_field is None:
        raise click.BadParameter("--position=target requires a target field. Model was trained without --target-field.")

    # create model and load weights
    match config.model.guardrails:
        case GuardrailsMethod.STRUCTURE_AND_VALUES:
            if not config.pipeline.path_in_field_tokens:
                raise Exception("GuardrailsMethod.STRUCTURE_AND_VALUES requires path_in_field_tokens=True")
            vpda = ObjectVPDA(encoder, schema)
        case GuardrailsMethod.STRUCTURE_ONLY:
            vpda = ObjectVPDA(encoder)
        case GuardrailsMethod.NONE:
            vpda = None

    model = ORIGAMI(config.model, config.train, vpda=vpda)
    model.load_state_dict(state_dict)

    # data configs
    config.data.source = source
    config.data.db = kwargs["source_db"]
    config.data.coll = kwargs["source_coll"]
    config.data.projection = create_projection(kwargs["include_fields"], kwargs["exclude_fields"])
    config.data.skip = kwargs["skip"]
    config.data.limit = kwargs["limit"]

    # validate MongoDB connection
    if source.startswith("mongodb://") or source.startswith("mongodb+srv://"):
        if kwargs["source_db"] is None or kwargs["source_coll"] is None:
            raise click.BadParameter("--source-db and --source-coll are required for MongoDB URI")

    # load data
    df = load_data(source, config.data)

    # validate non-empty dataset
    if len(df) == 0:
        raise click.ClickException("No documents loaded from source")

    # transform data using test pipeline
    test_pipeline = pipelines["test"]
    test_df = test_pipeline.transform(df)

    # create dataset
    test_dataset = DFDataset(test_df)

    if kwargs["verbose"]:
        # report number of parameters
        n_params = count_parameters(model)
        click.echo(f"running on device: {model.device}", err=True)
        click.echo(f"number of parameters: {n_params / 1e6:.2f}M", err=True)
        click.echo(f"loaded {len(test_dataset)} documents", err=True)
        click.echo(f"config:\n {OmegaConf.to_yaml(config)}", err=True)

    # initialize embedder
    embedder = Embedder(
        model,
        encoder,
        config.data.target_field,
        batch_size=config.train.batch_size,
    )

    # generate embeddings with progress bar
    if kwargs["verbose"]:
        click.echo("generating embeddings...", err=True)

    with tqdm(total=len(test_dataset), disable=not kwargs["verbose"], desc="Embedding", unit="doc") as pbar:
        # custom wrapper to update progress bar
        original_model_call = model.forward

        def forward_with_progress(*args, **fwd_kwargs):
            result = original_model_call(*args, **fwd_kwargs)
            pbar.update(args[0].size(0))  # update by batch size
            return result

        model.forward = forward_with_progress

        embeddings = embedder.embed(
            test_dataset,
            kwargs["position"],
            kwargs["reduction"],
        )

        model.forward = original_model_call

    # move to CPU
    embeddings = embeddings.cpu()

    # optional L2 normalization
    if kwargs["normalize"]:
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        embeddings = embeddings / norms.clamp(min=1e-12)

    # infer format from file extension
    output_path = pathlib.Path(kwargs["output_file"])

    # validate output directory exists
    if not output_path.parent.exists():
        raise click.BadParameter(f"Output directory does not exist: {output_path.parent}")

    try:
        format = infer_output_format(output_path)
    except ValueError as e:
        raise click.BadParameter(str(e))

    # save embeddings
    save_embeddings(embeddings.numpy(), output_path, format)

    if kwargs["verbose"]:
        click.echo(f"\nGenerated embeddings: shape={tuple(embeddings.shape)}", err=True)
        click.echo(f"Position={kwargs['position']}, Reduction={kwargs['reduction']}", err=True)
        click.echo(f"Normalized={kwargs['normalize']}", err=True)
        click.echo(f"Saved to: {output_path} (format: {format})", err=True)
    else:
        click.echo(f"Saved embeddings to: {output_path}")
