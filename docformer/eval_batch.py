import re

import evaluate
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def predict_labels_batch(texts, model, tokenizer, batch_size):
    all_predictions = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]

        # Truncate and prepare input
        trunc_texts = [re.sub(r'("target":).*', r"\1", text) for text in batch_texts]
        inputs = tokenizer(trunc_texts, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, num_return_sequences=1)

        # Decode predictions
        decoded_outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        # Extract labels from predictions
        batch_predictions = []
        for pred in decoded_outputs:
            matches = re.findall(r'"([^"]*)"', pred)
            batch_predictions.append(matches[0] if len(matches) > 0 else "")

        all_predictions.extend(batch_predictions)

    return all_predictions


def evaluate_gpt2_classification(model_path: str, dataset: Dataset, batch_size=32):
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(tokenizer.padding_side)
    model.eval()

    # Load metric
    accuracy_metric = evaluate.load("accuracy")

    # Make predictions in batches
    predictions = predict_labels_batch(dataset["doc"], model, tokenizer, batch_size)

    # Convert string predictions to integers
    def label_to_id(label):
        try:
            return dataset.features["target"].str2int(label)
        except ValueError:
            return -1

    predictions_int = [label_to_id(pred) for pred in predictions]

    # Compute metrics
    results = accuracy_metric.compute(predictions=predictions_int, references=dataset["target"])

    return results
