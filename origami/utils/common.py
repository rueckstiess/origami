import itertools
import operator
import pickle
import random
from collections import OrderedDict
from enum import Enum
from types import NoneType
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from origami.utils.config import TopLevelConfig

from .guild import print_guild_scalars


class QueryRegionEmptyException(Exception):
    pass


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


def sort_dict_fields(doc: dict) -> OrderedDict:
    """Sort dictionary fields alphabetically and return an OrderedDict.

    This ensures consistent field ordering across documents, which is important
    for models that are sensitive to field order (e.g., when tokenizing documents).

    Args:
        doc: Dictionary to sort

    Returns:
        OrderedDict with fields sorted alphabetically by key

    Example:
        >>> doc = {'z': 1, 'a': 2, 'b': 3}
        >>> sorted_doc = sort_dict_fields(doc)
        >>> list(sorted_doc.keys())
        ['a', 'b', 'z']
    """
    return OrderedDict(sorted(doc.items(), key=lambda x: x[0]))


def try_compare(x, op, y, default=False):
    try:
        return op(x, y)
    except TypeError:
        # if the types are not comparable, it's not a match
        # MongoDB calls this type bracketing
        return default


def eq_with_nan(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    return operator.eq(a, b)


def ge_with_nan(a, b):
    if eq_with_nan(a, b):
        return True
    return operator.ge(a, b)


def le_with_nan(a, b):
    if eq_with_nan(a, b):
        return True
    return operator.le(a, b)


def type_eq(a, b):
    TYPE_CLASSES = {
        "double": float,
        "string": str,
        "object": Symbol.SUBDOC_START,
        "array": ArrayStart,
        "bool": bool,
        "null": NoneType,
        "int": int,
        "number": (int, float, complex),
    }

    return isinstance(a, TYPE_CLASSES[b])


def size_eq(a, b):
    if isinstance(a, ArrayStart):
        return a.size == b
    return False


class EncodingTypes(Enum):
    ANY = 1
    CONSTANT = 2
    BINARY = 3
    CATEGORICAL = 4
    DATETIME = 5
    GAUSSIAN = 6
    MIXOFGAUSS = 7
    HISTOGRAM = 8
    MIXED = 9
    DOCUMENT = 10
    SET = 11


OPERATORS = {
    "gt": lambda a, val: try_compare(a, operator.gt, val[0]),
    "lt": lambda a, val: try_compare(a, operator.lt, val[0]),
    "gte": lambda a, val: try_compare(a, ge_with_nan, val[0]),
    "lte": lambda a, val: try_compare(a, le_with_nan, val[0]),
    "eq": lambda a, val: try_compare(a, eq_with_nan, val[0]),
    "ne": lambda a, val: try_compare(a, operator.ne, val[0], True),
    "in": lambda a, val: try_compare(val, operator.contains, a),
    "nin": lambda a, val: try_compare(val, lambda x, y: operator.not_(operator.contains(x, y)), a, True),
    "type": lambda a, val: try_compare(a, type_eq, val[0]),
    "size": lambda a, val: try_compare(a, size_eq, val[0]),
    "exists": lambda a, val: (a is not None and not pd.isna(a)) and val[0],
}


PYTHON_BSON_TYPE_MAP = {
    "str": "string",
    "int": "int",
    "float": "double",
    "bool": "bool",
    "datetime": "date",
    "ObjectId": "objectId",
    "dict": "object",
    "list": "array",
    "NoneType": "null",
}


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


def auto_device(device: str = "auto"):
    """utility function to determine automatically which device to run on."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    return device


def count_parameters(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def save_origami_model(model: torch.nn.Module, pipelines: dict, config: OmegaConf, path: str):
    model_dict = {
        "state_dict": model.state_dict(),
        "config": OmegaConf.to_container(config, enum_to_str=True),
        "pipelines": pipelines,
    }

    pickle.dump(model_dict, open(path, "wb"))


def load_origami_model(path: str):
    model_dict = pickle.load(open(path, "rb"))

    # re-validate config
    tlc = OmegaConf.create(TopLevelConfig)
    model_dict["config"] = OmegaConf.merge(tlc, model_dict["config"])

    return model_dict


def mult_error(estimate: int, actual: int) -> float:
    """calculates the multiplicative error between estimates and ground truth,
    as used in e.g. Naru."""

    # round to nearest integer and clamp at minimum of 1
    actual = max(1, np.round(actual))
    estimate = max(1, np.round(estimate))

    return max(actual, estimate) / min(actual, estimate)


def make_progress_callback(
    train_config: "TrainConfig",  # noqa: F821 # type: ignore
    train_dataset: Optional["DFDataset"] = None,  # noqa: F821 # type: ignore
    test_dataset: Optional["DFDataset"] = None,  # noqa: F821 # type: ignore
    predictor: Optional["Predictor"] = None,  # noqa: F821 # type: ignore
) -> Callable:
    predict_accuracy = hasattr(predictor, "accuracy")

    def progress_callback(model):
        if model.batch_num % train_config.print_every == 0:
            scalars = dict(
                step=f"{int(model.batch_num / train_config.print_every)}",
                epoch=model.epoch_num,
                batch_num=model.batch_num,
                batch_dt=f"{model.batch_dt * 1000:.2f}",
                batch_loss=f"{model.loss:.4f}",
                lr=f"{model.learning_rate:.2e}",
            )
            if model.batch_num % train_config.eval_every == 0:
                if train_dataset and train_config.sample_train > 0 and predict_accuracy:
                    # evaluate on a sample of the training data
                    scalars.update(
                        train_acc=f"{predictor.accuracy(train_dataset.sample(n=train_config.sample_train)):.4f}",
                    )
                if test_dataset and train_config.sample_test > 0:
                    # evaluate on a sample of the test data
                    scalars.update(
                        test_loss=f"{predictor.ce_loss(test_dataset.sample(n=train_config.sample_test)):.4f}",
                    )
                    if predict_accuracy:
                        scalars.update(
                            test_acc=f"{predictor.accuracy(test_dataset.sample(n=train_config.sample_test)):.4f}",
                        )

            print_guild_scalars(**scalars)

    return progress_callback


def parse_path(path: str) -> List[str]:
    """Split dot notation path into components."""
    return path.split(".")


def get_value_at_path(d: dict, path: List[str]) -> Tuple[Any, bool]:
    """Retrieve value at specified path in nested dictionary.
    Returns tuple of (value, found) where found is False if path doesn't exist."""
    current = d
    for component in path:
        if not isinstance(current, dict) or component not in current:
            return None, False
        current = current[component]
    return current, True


def reorder_with_target_last(d: dict, target_path: str) -> Tuple[OrderedDict, Any]:
    """
    Reorder dictionary so target field appears last, maintaining nested structure.
    Creates missing intermediate paths and sets target to Symbol.UNKNOWN if not found.
    """
    path_components = parse_path(target_path)
    target_value, found = get_value_at_path(d, path_components)

    def reorder_level(current_dict: dict, remaining_path: List[str]) -> OrderedDict:
        if not remaining_path:
            return OrderedDict(current_dict)

        current_target = remaining_path[0]
        result = OrderedDict()

        # Add all non-target fields first
        for k, v in current_dict.items():
            if k != current_target:
                result[k] = v if not isinstance(v, dict) else reorder_level(v, [])

        # Handle the target path
        if current_target in current_dict:
            target_dict = current_dict[current_target]
        else:
            # Create empty dict for missing intermediate paths
            target_dict = {} if len(remaining_path) > 1 else Symbol.UNKNOWN

        if len(remaining_path) > 1:
            # If we have more path components, recurse with remaining path
            result[current_target] = reorder_level(target_dict, remaining_path[1:])
        else:
            # If this is the final path component, add it last
            result[current_target] = (
                target_dict if not isinstance(target_dict, dict) else reorder_level(target_dict, [])
            )

        return result

    # Check if we're trying to traverse through a non-dict value
    current = d
    for i, component in enumerate(path_components[:-1]):
        if component in current and not isinstance(current[component], dict):
            # If we hit a non-dict value in the path, treat the entire remaining path
            # as a top-level field
            new_target = ".".join(path_components[i:])
            result = OrderedDict()
            for k, v in d.items():
                if k != new_target:
                    result[k] = v if not isinstance(v, dict) else reorder_level(v, [])
            result[new_target] = Symbol.UNKNOWN
            return result, Symbol.UNKNOWN

        if component not in current:
            break
        current = current[component]

    return reorder_level(d, path_components), target_value if found else Symbol.UNKNOWN


# def reorder_with_target_last(d: dict, target_path: str) -> Tuple[OrderedDict, Any]:
#     """
#     Reorder dictionary so target field appears last, maintaining nested structure.
#     If target field doesn't exist, returns (OrderedDict(d), Symbol.UNKNOWN).
#     """
#     path_components = parse_path(target_path)
#     target_value, found = get_value_at_path(d, path_components)

#     if not found:
#         od = OrderedDict(d)
#         od[target_path] = Symbol.UNKNOWN
#         return od, Symbol.UNKNOWN

#     def reorder_level(current_dict: dict, remaining_path: List[str]) -> OrderedDict:
#         if not remaining_path:
#             return OrderedDict(current_dict)

#         current_target = remaining_path[0]
#         result = OrderedDict()

#         # Add all non-target fields first
#         for k, v in current_dict.items():
#             if k != current_target:
#                 result[k] = v if not isinstance(v, dict) else reorder_level(v, [])

#         # Add target field last
#         if current_target in current_dict:
#             target_dict = current_dict[current_target]
#             if len(remaining_path) > 1:
#                 # If we have more path components, recurse with remaining path
#                 result[current_target] = reorder_level(target_dict, remaining_path[1:])
#             else:
#                 # If this is the final path component, add it last
#                 result[current_target] = (
#                     target_dict if not isinstance(target_dict, dict) else reorder_level(target_dict, [])
#                 )

#         return result

#     return reorder_level(d, path_components), target_value
