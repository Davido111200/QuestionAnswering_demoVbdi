import torch
import torch.nn as nn
from tqdm import tqdm

from collections import Counter
import string
import re

def evaluate(model, data_loader, device):
    all_results = []
    model.eval()
    for batch in tqdm(data_loader):
        with torch.no_grad():
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'token_type_ids': batch['token_type_ids'].to(device),
                'start_positions': None,
                'end_positions': None,
                'is_impossibles': None,
                'position_ids': batch['position_ids'].to(device),
            }

            outputs = model(**inputs)
            all_results.append(outputs)

    exact_match, f1 = calculate_metrics(data_loader.data, all_results)
    print('exact_match: {}, f1: {}'.format(exact_match, f1))
    return exact_match, f1


def calculate_metrics(dataset, results):
    exact_match = 0
    f1 = 0
    for sample, result in zip(dataset, results):
        ground_truths = sample.answers
        prediction = {
            "text": sample.context_text[result[1].argmax(): result[2].argmax()],
            "answer_start": result[1].argmax(),
            "answer_end": result[2].argmax()
        }
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths
        )
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths
        )
    exact_match = 100.0 * exact_match / len(dataset)
    f1 = 100.0 * f1 / len(dataset)
    return exact_match, f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction["text"]) == normalize_answer(ground_truth["text"]))

def f1_score(prediction, ground_truths):
    return metric_max_over_ground_truths(
        f1_score_for_tokens, prediction, ground_truths
    )

def f1_score_for_tokens(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction["text"]).split()
    ground_truth_tokens = normalize_answer(ground_truth["text"]).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))