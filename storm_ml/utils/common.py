import itertools
import json
import os
import random
import sys
from collections import OrderedDict
from enum import Enum
from typing import Any, Generator, Iterable, Optional, Tuple, Union

import numpy as np
import torch


class Symbol(Enum):
    PAD = 0
    START = 1
    END = 2
    FIELD = 3
    VALUE = 4
    DOC = 5
    SUBDOC_START = 6
    SUBDOC_END = 7
    ARRAY_END = 8
    UNKNOWN = 9


class FieldToken:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f'<Field "{self.name}">'

    def __eq__(self, other):
        if isinstance(other, FieldToken):
            return self.name == other.name
        else:
            return False

    def __hash__(self):
        return hash(self.name)


class ArrayStart:
    def __init__(self, size: int):
        self._size = size

    @property
    def size(self):
        return self._size

    def __repr__(self) -> str:
        return f"<ArrayStart {str(self.size)}>"

    def __eq__(self, other):
        if isinstance(other, ArrayStart):
            return self.size == other.size
        else:
            return False

    def __hash__(self):
        return hash(self.size)


def pad_trunc(lst, length, pad_token=Symbol.PAD):
    """pads or truncates a list so that is has length. If the original list was shorter,
    it fills the end of the list with `pad_id` values."""
    if len(lst) > length:
        return lst[:length]
    if len(lst) < length:
        return lst + [pad_token] * (length - len(lst))
    return lst


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# def setup_logging(config):
#     """monotonous bookkeeping"""
#     work_dir = config.system.work_dir
#     # create the work directory if it doesn't already exist
#     os.makedirs(work_dir, exist_ok=True)
#     # log the args (if any)
#     with open(os.path.join(work_dir, "args.txt"), "w") as f:
#         f.write(" ".join(sys.argv))
#     # log the config itself
#     with open(os.path.join(work_dir, "config.json"), "w") as f:
#         f.write(json.dumps(config.to_dict(), indent=4))


# def permute_document(doc):
#     """shuffle the keys and values in a dictionary"""
#     items = list(doc.items())
#     random.shuffle(items)
#     return OrderedDict(items)


def torch_isin(input: torch.Tensor, allowed: list) -> torch.Tensor:
    """returns a boolean mask of the same shape as input, where True indicates that
    the value is one of the allowed values.

    This is a workaround for torch.isin(), which is not supported on MPS devices yet.

    Example:
        input = tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

        allowed = tensor([2, 4, 6, 8])

        result = tensor([[False, True,  False],
                         [True,  False, True],
                         [False, True,  False]])

    """
    if len(allowed) == 0:
        return torch.zeros(input.size(), dtype=torch.bool, device=input.device)
    mask = input == allowed[0]
    for a in allowed[1:]:
        mask = mask | (input == a)
    return mask


def peek_generator(iterable: Iterable[Any]) -> Tuple[Any, Generator[Any, Any, Any]]:
    """utility to peek at the first value of a generator without consuming it."""
    try:
        peek_value = next(iterable)
        return peek_value, itertools.chain([peek_value], iterable)
    except StopIteration:
        # Handle the case where the generator is empty.
        return None, iterable


from dataclasses import dataclass, field
from enum import Enum

from omegaconf import MISSING, OmegaConf


class EstimationMethod(Enum):
    SAMPLING = 1
    FIELD_PROMPTING = 2


class SequenceOrderMethod(Enum):
    ORDERED = 1
    SHUFFLED = 2


class NumericMethod(Enum):
    BINNING = 1


class PositionEncodingMethod(Enum):
    NONE = 1
    INTEGER = 2
    DOCUMENT = 3


class BaseConfig:
    """Wrap all configs deriving from BaseConfig in OmegaConf objects."""

    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        cls.__init__(instance, **kwargs)
        return OmegaConf.structured(instance)


@dataclass(kw_only=True)
class ModelConfig(BaseConfig):
    """Model config for creating a GPT model."""

    # architecture parameters
    n_layer: int = MISSING
    n_head: int = MISSING
    n_embd: int = MISSING
    vocab_size: int = MISSING
    block_size: int = MISSING

    # dropout hyperparameters
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

    # weight tieing between embedding matrix and linear head weights (saves parameters)
    tie_weights: bool = False

    # the default position encoding method to use (NONE, INTEGER, DOCUMENT)
    position_encoding: PositionEncodingMethod = MISSING

    # Whether to use an MLP to fuse position and token embeddings or just sum them
    fuse_pos_with_mlp: bool = False

    # whether or not to mask field tokens in loss calculation
    mask_field_token_losses: bool = False

    # whether or not to use guardrails (requires a DocumentVPDA to be passed into model)
    guardrails: bool = True

    @staticmethod
    def from_preset(size: str, **kwargs) -> "ModelConfig":
        match size:
            case "gpt-nano":
                return ModelConfig(n_layer=3, n_head=3, n_embd=48, **kwargs)
            case "gpt-micro":
                return ModelConfig(n_layer=4, n_head=4, n_embd=128, **kwargs)
            case "gpt-mini":
                return ModelConfig(n_layer=6, n_head=6, n_embd=192, **kwargs)
            case "gpt-small":
                return ModelConfig(n_layer=8, n_head=8, n_embd=384, **kwargs)
            case "openai-gpt":
                return ModelConfig(n_layer=12, n_head=12, n_embd=768, **kwargs)
            case _:
                raise ValueError(f"Unknown model size: {size}")


@dataclass(repr=True, kw_only=True)
class TrainConfig(BaseConfig):
    """Train config for training a GPT model."""

    # device to train on (auto, cpu, cuda, mps)
    device: str = "auto"

    # dataloder parameters (currently not used)
    num_workers: int = 0

    # optimizer parameters
    batch_size: int = 100
    n_batches: int = MISSING
    n_warmup_batches: int = 1000
    learning_rate: float = 1e-3
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1  # only applied on matmul weights
    grad_norm_clip: float = 1.0
    lr_end_factor: float = 0.01

    # print and eval options
    eval_every: int = 100
    sample_eval: int = 100
    sample_test: int = 100


@dataclass(repr=True, kw_only=True)
class PipelineConfig(BaseConfig):
    """Config needed to build and execute pipelines."""

    # split options
    test_split: float = MISSING
    shuffle_split: bool = True

    # pipeline options
    n_bins: int = 100
    sequence_order: SequenceOrderMethod = MISSING
    upscale: int = 1


@dataclass(repr=True, kw_only=True)
class DataConfig(BaseConfig):
    """Config to define data source and target field."""

    # database parameters
    db: str = MISSING
    coll: str = MISSING
    projection: dict = field(default_factory=lambda: {"_id": 0})
    limit: int = 0

    # prediction options
    target_field: str = MISSING


@dataclass(repr=True, kw_only=True)
class TopLevelConfig(BaseConfig):
    """combined Model/Train/Data/Pipeline config."""

    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()
    pipeline: PipelineConfig = PipelineConfig()


def walk_all_leaf_kvs(
    item,
    path="",
    parent: Optional[Union[list, dict, OrderedDict]] = None,
    idx: Union[int, str] = None,
    key: str = None,
    pos: int = None,
    include_pos_in_path: bool = False,
):
    """
    Recursively walks through a nested dictionary or list structure and yields all the leaf key-value pairs.

    Args:
        item (Union[dict, OrderedDict, list]): The item to walk through.
        path (str, optional): The current path in the nested structure. Defaults to "".
        parent (Any, optional): The parent of the current item. Defaults to None.
        idx (Union[int, None], optional): The index of the current item in the list. Defaults to None.
        include_pos_in_path (bool, optional): Whether to include the position of the item in the path. Defaults to False.
            if True: {foo: [{bar: "x"}]} will produce path "foo.[0].bar" for value "x"
            if False: {foo: [{bar: "x"}]} will produce path "foo.[].bar" for value "x"

    Yields:
        dict: A dictionary with the following keys:
            - parent (Any): The parent of the current item.
            - idx (Union[int, str]): The index of the current item in the parent. int for lists, str for dicts.
            - key (str): The key of the current dictionary item.
            - pos (int): The position of the current list item.
            - value (Any): The value of the current item.
            - path (str): The current path in the nested structure.
    """
    if isinstance(item, (dict, OrderedDict)):
        for key, value in item.items():
            new_path = f"{path}.{key}".lstrip(".")
            yield from walk_all_leaf_kvs(
                value, path=new_path, parent=item, idx=key, key=key, pos=pos, include_pos_in_path=include_pos_in_path
            )
    elif isinstance(item, list):
        for pos, value in enumerate(item):
            if include_pos_in_path:
                new_path = f"{path}.[{pos}]".lstrip(".")
            else:
                new_path = f"{path}.[]".lstrip(".")
            yield from walk_all_leaf_kvs(
                value,
                path=new_path,
                parent=item,
                idx=pos,
                pos=pos,
                key=key,
                include_pos_in_path=include_pos_in_path,
            )
    else:
        yield {"parent": parent, "idx": idx, "key": key, "pos": pos, "value": item, "path": path}


def flatten_docs(docs: list[dict]) -> list[dict]:
    flat = [{item["path"]: item["value"] for item in walk_all_leaf_kvs(doc, include_pos_in_path=True)} for doc in docs]

    return flat
