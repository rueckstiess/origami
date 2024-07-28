import json

from datasets import ClassLabel, Dataset, Features, Value
from pymongo import MongoClient
from transformers import PreTrainedTokenizerBase


def load_from_mongodb(
    uri: str,
    db_name: str,
    coll_name: str,
    target_field: str,
    test_size: float = 0.2,
    as_dict: bool = False,
    **kwargs,
) -> Dataset:
    # create MongoDB client and collection
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[coll_name]

    # get labels
    labels = collection.distinct(target_field)

    # define the features of our dataset
    features = Features(
        {
            "doc": Value("string"),
            "target": ClassLabel(names=labels),
        }
    )

    # get documents, project out _id
    docs = list(collection.find(projection={"_id": 0}))
    for doc in docs:
        target = doc["target"]
        del doc["target"]
        doc["target"] = target

    # create dataset
    dataset = Dataset.from_dict(
        {
            "doc": [d if as_dict else json.dumps(d) for d in docs],
            "target": [d[target_field] for d in docs],
        },
        features=features,
    )

    split = dataset.train_test_split(
        test_size=test_size,
        seed=kwargs.get("seed", 42),
        shuffle=kwargs.get("shuffle", True),
    )
    train_dataset = split["train"]
    test_dataset = split["test"]

    if kwargs.get("verbose", False):
        print(f"Full dataset size: {len(dataset)} samples")
        print(f"Train dataset size: {len(train_dataset)} samples")
        print(f"Test dataset size: {len(test_dataset)} samples")

    return train_dataset, test_dataset


def tokenize(
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    max_length: int = 128,
    batch_size: int = 1000,
) -> Dataset:
    def tokenize_function(examples):
        return tokenizer(
            examples["doc"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    return tokenized
