import json
import re

import evaluate
import torch
from datasets import ClassLabel, Dataset, Features, Value
from pymongo import MongoClient
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizerBase


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


def predict_label(text, model, tokenizer):
    trunc_text = re.sub(r'("target":).*', r"\1", text)
    input_ids = tokenizer.encode(trunc_text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=5, num_return_sequences=1)
    pred = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)

    # Find label in prediction
    matches = re.findall(r'"([^"]*)"', pred)
    return matches[0] if len(matches) > 0 else ""


def evaluate_gpt2_classification(model_path: str, dataset: Dataset):
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    # Load metric
    accuracy_metric = evaluate.load("accuracy")

    # Make predictions
    predictions = [predict_label(text, model, tokenizer) for text in tqdm(dataset["doc"])]
    print(predictions)
    print(dataset["target"])

    # Convert string predictions to integers
    def label_to_id(label):
        try:
            return dataset.features["target"].str2int(label)
        except ValueError:
            print("Invalid label:", label)
            return -1

    predictions_int = [label_to_id(pred) for pred in predictions]

    # Compute metrics
    results = accuracy_metric.compute(predictions=predictions_int, references=dataset["target"])

    return results
