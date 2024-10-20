import itertools
import random
from collections import OrderedDict
from enum import Enum
from typing import Any, Generator, Iterable, Optional, Tuple, Union
from omegaconf import OmegaConf
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


def save_model(model: torch.nn.Module, config: OmegaConf, path: str):
    model_dict = {"state_dict": model.state_dict()} | {"config": OmegaConf.to_container(config, enum_to_str=True)}
    torch.save(model_dict, path)


def load_model(path: str):
    model_dict = torch.load(path)
    config = OmegaConf.create(model_dict.pop("config"))

    return model_dict, config
