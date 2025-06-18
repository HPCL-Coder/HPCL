import json
from typing import List, Tuple, Dict
from collections import defaultdict
import re
from difflib import SequenceMatcher
import numpy as np
import statistics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import matplotlib.pyplot as plt

def evaluate_line_level(preds, gts):
    pred_lines = [line for line, _, _ in preds]
    gt_lines = [line for line, _, _ in gts]
    all_lines = sorted(set(pred_lines + gt_lines))

    y_pred = [1 if l in pred_lines else 0 for l in all_lines]
    y_true = [1 if l in gt_lines else 0 for l in all_lines]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)


def evaluate_typeAline_level(preds, gts):
    pred_type_set = set([(line, typ) for line, typ, _ in preds])
    gt_type_set = set([(line, typ) for line, typ, _ in gts])
    all_pairs = sorted(pred_type_set | gt_type_set)

    y_pred = [1 if p in pred_type_set else 0 for p in all_pairs]
    y_true = [1 if p in gt_type_set else 0 for p in all_pairs]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1

def evaluate_type_level_ignore_line(preds, gts):
    """
    Compare types while ignoring line numbers.
    This treats the task as multi-label classification of unordered parallel directives.
    """
    pred_types = [ptype for _, ptype, _ in preds]
    gt_types = [gtype for _, gtype, _ in gts]
    all_types = sorted(set(pred_types + gt_types))

    y_pred = [pred_types.count(t) for t in all_types]
    y_true = [gt_types.count(t) for t in all_types]

    # Convert to binary presence/absence for precision/recall
    y_pred_bin = [1 if c > 0 else 0 for c in y_pred]
    y_true_bin = [1 if c > 0 else 0 for c in y_true]

    precision = precision_score(y_true_bin, y_pred_bin)
    recall = recall_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)

    return precision, recall, f1

def evaluate_overall(preds, gts):
    """
    Compare types while ignoring line numbers.
    This treats the task as multi-label classification of unordered parallel directives.
    """

    all_results = sorted(set(preds + gts))

    y_pred = [preds.count(t) for t in all_results]
    y_true = [gts.count(t) for t in all_results]

    # Convert to binary presence/absence for precision/recall
    y_pred_bin = [1 if c > 0 else 0 for c in y_pred]
    y_true_bin = [1 if c > 0 else 0 for c in y_true]

    precision = precision_score(y_true_bin, y_pred_bin)
    recall = recall_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)

    return precision, recall, f1

from nltk.translate.bleu_score import corpus_bleu

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def evaluate_code_bleu_corpus_flexible(preds, gts,  smoothing=True):
    """
    Flexible corpus BLEU evaluator.
    Supports BLEU-1, BLEU-2, BLEU-4 with optional smoothing.
    
    Args:
        preds: list of (line, type, code)
        gts: list of (line, type, code)
        max_ngram: int, maximum n-gram to use (1~4)
        smoothing: bool, whether to use smoothing
        
    Returns:
        float: corpus-level BLEU score
    """
    pred_codes = [code for _, _, code in preds]
    gt_codes = [code for _, _, code in gts]

    hypotheses = [pred.split() for pred in pred_codes]
    references = [[gt.split()] for gt in gt_codes]

    # Pad to equal length
    max_len = max(len(hypotheses), len(references))
    min_len = min(len(hypotheses), len(references))
    while len(hypotheses) < max_len:
        hypotheses.append(["<pad>"])
    while len(references) < max_len:
        references.append([["<pad>"]])

    # Define n-gram weights dynamically
    max_ngram = max(1, min(min_len, 4))  # clamp to [1, 4]
    weights = tuple([1.0 / max_ngram] * max_ngram + [0.0] * (4 - max_ngram))

    smoother = SmoothingFunction().method1 if smoothing else None
    score = corpus_bleu(references, hypotheses, weights=weights, smoothing_function=smoother)
    return score

def evaluate_code_bleu_corpus(preds, gts):
    """
    Evaluate BLEU score using corpus-level BLEU.
    Suitable for evaluating overall token-level overlap in multi-line code generation.
    """
    pred_codes = [code for _, _, code in preds]
    gt_codes = [code for _, _, code in gts]

    # Tokenize
    hypotheses = [pred.split() for pred in pred_codes]
    references = [[gt.split()] for gt in gt_codes]  # reference must be list of lists

    # Pad to equal length: each hypothesis must have a corresponding reference
    max_len = max(len(hypotheses), len(references))
    while len(hypotheses) < max_len:
        hypotheses.append(["<pad>"])
    while len(references) < max_len:
        references.append([["<pad>"]])

    score = corpus_bleu(references, hypotheses)
    return score


def evaluate_multi_instruction(preds, gts):
    line_precision, line_recall, line_f1 = evaluate_line_level(preds, gts)
    type_precision, type_recall, type_f1 = evaluate_typeAline_level(preds, gts)
    type_only_precision, type_only_recall, type_only_f1 = evaluate_type_level_ignore_line(preds, gts)
    overall_precision, overall_recall, overall_f1 = evaluate_overall(preds, gts)
    code_bleu_scores_multimatch_flexible = evaluate_code_bleu_corpus_flexible(preds, gts)
    # code_bleu_scores_multimatch  = evaluate_code_bleu_corpus(preds, gts)
    return {
        "line_precision": line_precision,
        "line_recall": line_recall,
        "line_f1": line_f1,
        "type_precision": type_precision,
        "type_recall": type_recall,
        "type_f1": type_f1,
        "type_only_precision": type_only_precision,
        "type_only_recall": type_only_recall,
        "type_only_f1": type_only_f1,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "code_bleu_scores_multimatch_flexible": code_bleu_scores_multimatch_flexible,
    }


def parse_answer(text: str):
    """
    Parse text data in the format of "(line_number, parallel_type, code_line) (line_number, parallel_type, code_line)..."
    and return a list of tuples (line_number, parallel_type, code_line).
    Where:
      - line_number is the first field (try to convert to int)
      - parallel_type is the second field
      - the remaining fields are concatenated into code_line
    """
    if not text:
        return []

    # Remove leading and trailing whitespace
    text = text.strip()

    # If the front starts with "(", remove it
    if text.startswith("("):
        text = text[1:]
    # If the end is ")", also remove it
    if text.endswith(")"):
        text = text[:-1]
    
    # Split using ") (" as the delimiter
    # Now each part is like "5, OpenMP, #pragma omp parallel for"
    parts = text.split("); (")

    result = []
    for part in parts:
        # Split by comma and remove spaces
        fields = [p.strip() for p in part.split(",")]
        if len(fields) >= 2:
            line_number_str = fields[0]
            parallel_type = fields[1]

            # Concatenate the remaining fields into code_line
            if len(fields) > 2:
                code_line = ",".join(fields[2:]).strip()
            else:
                code_line = ""

            # Try to convert the line number to an integer
            try:
                line_number = int(line_number_str)
            except ValueError:
                # print("Unable to convert line_number_str", line_number_str)
                line_number = -1

            result.append((line_number, parallel_type, code_line))

    return result


def evaluate_predictions(results_file):
    """Evaluate the prediction results in the jsonl file and save the extracted answers"""
    rewards = []
    extracted_results = []
    predictions = []
    ground_truths = []
    
    # Dictionary for accumulating all metrics
    metrics_sum = {
        'line_precision': 0, 'line_recall': 0, 'line_f1': 0,
        'type_precision': 0, 'type_recall': 0, 'type_f1': 0,
        'type_only_precision': 0, 'type_only_recall': 0, 'type_only_f1': 0,
        "overall_precision": 0,
        "overall_recall": 0,
        "overall_f1": 0,
        'code_bleu_scores_multimatch_flexible': 0
    }
    valid_count = 0
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse the JSON of each line
            result = json.loads(line)
            
            # Get the prediction result and the true label
            prediction = result.get('predict', '')
            ground_truth = result.get('label', '')
            predictions.append(prediction)
            ground_truths.append(ground_truth)

    # Calculate evaluation metrics for each sample
    for pred, gt in zip(predictions, ground_truths):
        # Parse the prediction and the true label
        pred_tuples = parse_answer(pred)
        gt_tuples = parse_answer(gt)
        
        if pred_tuples and gt_tuples:
            # Calculate all metrics for this sample
            scores = evaluate_multi_instruction(pred_tuples, gt_tuples)
            
            # Accumulate all metrics
            for metric in metrics_sum:
                if metric in scores and scores[metric] is not None:
                    metrics_sum[metric] += scores[metric]
            
            valid_count += 1

    # Calculate the average
    metrics_avg = {}
    for metric in metrics_sum:
        metrics_avg[metric] = metrics_sum[metric] / len(predictions) if len(predictions) > 0 else 0
    
    return {
        'metrics_avg': metrics_avg,
        'total_samples': len(predictions),
        'valid_samples': valid_count
    }

def evaluate_predictions_by_type(results_file):
    """Evaluate the prediction results in the jsonl file by parallel type"""
    predictions = []
    ground_truths = []
    
    # Read the predictions and the true labels
    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            prediction = result.get('predict', '')
            ground_truth = result.get('label', '')
            predictions.append(prediction)
            ground_truths.append(ground_truth)
    
    # Accumulate metrics grouped by parallel type
    type_metrics = defaultdict(lambda: {
        'count': 0,
        'line_precision': 0, 'line_recall': 0, 'line_f1': 0,
        'type_precision': 0, 'type_recall': 0, 'type_f1': 0,
        'type_only_precision': 0, 'type_only_recall': 0, 'type_only_f1': 0,
        'code_bleu_scores_multimatch_flexible': 0
    })
    
    # Calculate evaluation metrics for each sample
    for pred, gt in zip(predictions, ground_truths):
        pred_tuples = parse_answer(pred)
        gt_tuples = parse_answer(gt)
        
        if not pred_tuples or not gt_tuples:
            continue
        
        # Get the parallel types in this sample
        parallel_types = set(t[1] for t in gt_tuples)
        
        # Calculate all metrics for this sample
        scores = evaluate_multi_instruction(pred_tuples, gt_tuples)
        
        # Accumulate metrics by parallel type
        for p_type in parallel_types:
            # Filter out the tuples of the current type
            current_pred_tuples = [t for t in pred_tuples if t[1] == p_type]
            current_gt_tuples = [t for t in gt_tuples if t[1] == p_type]
            
            if current_pred_tuples or current_gt_tuples:
                # Calculate the metrics for the current type (even if the prediction is empty, let evaluate_multi_instruction handle it)
                type_scores = evaluate_multi_instruction(current_pred_tuples, current_gt_tuples)
                
                # Accumulate metrics
                for metric in type_metrics[p_type]:
                    if metric != 'count' and metric in type_scores and type_scores[metric] is not None:
                        type_metrics[p_type][metric] += type_scores[metric]
                
                type_metrics[p_type]['count'] += 1

    
    # Calculate the average for each type
    type_metrics_avg = {}
    for p_type, metrics in type_metrics.items():
        count = metrics['count']
        if count > 0:
            type_metrics_avg[p_type] = {
                'count': count
            }
            for metric in metrics:
                if metric != 'count':
                    type_metrics_avg[p_type][metric] = metrics[metric] / count
    
    return type_metrics_avg


from collections import defaultdict

def evaluate_predictions_by_label_count(results_file):
    """Evaluate the prediction results classified by the number of labels in the ground-truth"""
    predictions = []
    ground_truths = []
    
    # Read the predictions and the true labels
    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            prediction = result.get('predict', '')
            ground_truth = result.get('label', '')
            predictions.append(prediction)
            ground_truths.append(ground_truth)
    
    # Bucket statistics, the key is the number of labels
    label_count_metrics = defaultdict(lambda: {
        'count': 0,
        'line_precision': 0, 'line_recall': 0, 'line_f1': 0,
        'type_precision': 0, 'type_recall': 0, 'type_f1': 0,
        'type_only_precision': 0, 'type_only_recall': 0, 'type_only_f1': 0,
        'code_bleu_scores_multimatch_flexible': 0
    })

    for pred, gt in zip(predictions, ground_truths):
        pred_tuples = parse_answer(pred)
        gt_tuples = parse_answer(gt)

        if not gt_tuples:
            continue  # Skip if the ground-truth is empty

        label_count = len(gt_tuples)  # Number of labels in the ground-truth

        # Calculate the metrics for the current sample
        scores = evaluate_multi_instruction(pred_tuples, gt_tuples)

        for metric in label_count_metrics[label_count]:
            if metric != 'count' and metric in scores and scores[metric] is not None:
                label_count_metrics[label_count][metric] += scores[metric]

        label_count_metrics[label_count]['count'] += 1

    # Calculate the average for each label count bucket
    label_count_metrics_avg = {}
    for label_num, metrics in label_count_metrics.items():
        count = metrics['count']
        if count > 0:
            label_count_metrics_avg[label_num] = {'count': count}
            for metric in metrics:
                if metric != 'count':
                    label_count_metrics_avg[label_num][metric] = metrics[metric] / count

    return label_count_metrics_avg


import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate prediction results from a JSONL file.')
    parser.add_argument('results_file', type=str, help='Path to the JSONL file containing prediction results.')
    args = parser.parse_args()

    results_file = args.results_file
    # Evaluate the overall results
    eval_results = evaluate_predictions(results_file)

    print("=== Overall Evaluation Results ===")
    print(f"Total samples: {eval_results['total_samples']}")
    print(f"Valid samples: {eval_results['valid_samples']}")
    print("\nAverage values of each metric:")
    for metric, value in eval_results['metrics_avg'].items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()