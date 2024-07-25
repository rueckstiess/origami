from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset
import evaluate
import torch
import re
from tqdm.auto import tqdm


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
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model.eval()

    # Load metric
    accuracy_metric = evaluate.load("accuracy")

    # Make predictions
    predictions = [
        predict_label(text, model, tokenizer) for text in tqdm(dataset["doc"])
    ]

    # Convert string predictions to integers
    label_to_id = dataset.features["target"].str2int
    predictions_int = [label_to_id(pred) for pred in predictions]

    # Compute metrics
    results = accuracy_metric.compute(
        predictions=predictions_int, references=dataset["target"]
    )

    return results
