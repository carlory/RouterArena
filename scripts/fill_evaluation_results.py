#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Fill evaluation_result fields for deepseek-chat and deepseek-reasoner cached results.

This script uses the existing evaluation framework to compute:
- extracted_answer: Extracted from generated_answer using EnhancedAnswerExtractor
- ground_truth: Retrieved from dataset based on global_index
- score: Calculated using appropriate metric (mcq_accuracy, math_metric, etc.)
- metric: Determined by dataset type
- inference_cost: Calculated from token_usage and model pricing

Usage:
    python scripts/fill_evaluation_results.py
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import pandas as pd

# Add paths to import evaluation modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'llm_evaluation'))
sys.path.insert(0, project_root)

from llm_evaluation.enhanced_extractor import EnhancedAnswerExtractor
from llm_evaluation.metrics import mcq_accuracy, math_metric
from llm_evaluation.eval_reasoning import get_scorers_for_dataset


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(file_path: str, data: List[Dict[str, Any]]):
    """Save JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} entries to {file_path}")


def load_cost_config() -> Dict[str, Any]:
    """Load model cost configuration."""
    cost_path = os.path.join(project_root, 'model_cost', 'cost.json')
    if os.path.exists(cost_path):
        with open(cost_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def calculate_inference_cost(model_name: str, token_usage: Dict[str, int], cost_config: Dict[str, Any]) -> float:
    """
    Calculate inference cost based on token usage and model pricing.

    Args:
        model_name: Name of the model
        token_usage: Dict with input_tokens and output_tokens
        cost_config: Cost configuration from cost.json

    Returns:
        Cost in USD
    """
    if not token_usage or not cost_config:
        return 0.0

    # Normalize model name for lookup
    model_key = model_name.lower().replace('/', '-')

    # Try exact match first
    if model_key in cost_config:
        pricing = cost_config[model_key]
    else:
        # Try partial match
        matching_keys = [k for k in cost_config.keys() if model_key in k or k in model_key]
        if matching_keys:
            pricing = cost_config[matching_keys[0]]
        else:
            print(f"Warning: No pricing found for model {model_name}")
            return 0.0

    input_tokens = token_usage.get('input_tokens', 0)
    output_tokens = token_usage.get('output_tokens', 0)

    input_cost = (input_tokens / 1_000_000) * pricing.get('input_token_price_per_million', 0)
    output_cost = (output_tokens / 1_000_000) * pricing.get('output_token_price_per_million', 0)

    return input_cost + output_cost


def load_ground_truth_data() -> Dict[str, Any]:
    """Load all ground truth data from dataset."""
    from datasets import load_from_disk
    import pandas as pd

    ground_truth = {}
    dataset_dir = os.path.join(project_root, 'dataset', 'routerarena')

    if not os.path.exists(dataset_dir):
        print(f"Warning: Dataset directory not found at {dataset_dir}")
        return ground_truth

    try:
        # Load the router eval benchmark dataset (HuggingFace format)
        router_eval_bench = load_from_disk(dataset_dir)
        router_eval_bench_df = pd.DataFrame(router_eval_bench)

        # Build ground truth mapping: global_index -> answer
        for _, row in router_eval_bench_df.iterrows():
            global_index = row.get("Global Index") or row.get("global_index")
            answer = row.get("Answer") or row.get("answer")
            if global_index:
                ground_truth[global_index] = answer

        print(f"Loaded {len(ground_truth)} ground truth entries")
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        raise

    return ground_truth


def parse_dataset_from_global_index(global_index: str) -> str:
    """Extract dataset name from global_index (e.g., 'ArcMMLU_98' -> 'ArcMMLU')."""
    if '_' in global_index:
        return global_index.rsplit('_', 1)[0]
    return global_index


def get_metric_name_for_dataset(dataset_name: str) -> str:
    """Get the metric name for a dataset."""
    # Map dataset to metric name
    dataset_metric_map = {
        'AIME': 'math_metric',
        'MATH': 'math_metric',
        'ArcMMLU': 'mcq_accuracy',
        'GPQA': 'mcq_accuracy',
        'MMLU': 'mcq_accuracy',
        'MMLUPro': 'mcq_accuracy',
    }

    # Try exact match
    if dataset_name in dataset_metric_map:
        return dataset_metric_map[dataset_name]

    # Try partial match (for datasets like ArcMMLU_X)
    for key in dataset_metric_map:
        if key in dataset_name or dataset_name in key:
            return dataset_metric_map[key]

    # Default to mcq_accuracy for unknown datasets
    return 'mcq_accuracy'


def evaluate_entry(
    entry: Dict[str, Any],
    ground_truth_data: Dict[str, Any],
    cost_config: Dict[str, Any],
    extractor: EnhancedAnswerExtractor
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single entry and return the evaluation_result.

    Args:
        entry: The cache entry with generated_answer and metadata
        ground_truth_data: Dictionary mapping global_index to answers
        cost_config: Cost configuration
        extractor: Answer extractor instance

    Returns:
        evaluation_result dictionary or None if evaluation fails
    """
    global_index = entry.get('global_index')
    generated_answer = entry.get('generated_answer', '')
    token_usage = entry.get('token_usage', {})
    llm_selected = entry.get('llm_selected', '')

    if not global_index:
        print(f"Warning: Entry missing global_index")
        return None

    # Parse dataset name
    dataset_name = parse_dataset_from_global_index(global_index)

    # Get ground truth
    ground_truth = ground_truth_data.get(global_index)
    if ground_truth is None:
        print(f"Warning: No ground truth found for {global_index}")
        return None

    # Extract answer
    extracted_answer = extractor.extract_boxed_answer(generated_answer, dataset_name)

    # Get metric name
    metric_name = get_metric_name_for_dataset(dataset_name)

    # Calculate score based on metric
    score = 0.0
    try:
        if metric_name == 'mcq_accuracy':
            # MCQ: compare extracted answer with ground truth letter
            accuracy, raw_results = mcq_accuracy([extracted_answer], [ground_truth])
            score = float(raw_results[0].get('correct', False))
        elif metric_name == 'math_metric':
            # Math: use math_metric for comparison
            accuracy, raw_results = math_metric([generated_answer], [ground_truth])
            score = float(raw_results[0].get('correct', False))
        else:
            # Default: exact match
            score = 1.0 if str(extracted_answer).strip() == str(ground_truth).strip() else 0.0
    except Exception as e:
        print(f"Error calculating score for {global_index}: {e}")
        score = 0.0

    # Calculate inference cost
    inference_cost = calculate_inference_cost(llm_selected, token_usage, cost_config)

    # Build evaluation result
    evaluation_result = {
        'extracted_answer': extracted_answer,
        'ground_truth': ground_truth,
        'score': score,
        'metric': metric_name,
        'inference_cost': inference_cost
    }

    return evaluation_result


def fill_evaluation_results(file_path: str):
    """
    Fill evaluation_result fields for a cached results file.

    Args:
        file_path: Path to the JSONL cached results file
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    # Load data
    print("Loading cached results...")
    data = load_jsonl(file_path)
    print(f"Loaded {len(data)} entries")

    # Load ground truth
    ground_truth_data = load_ground_truth_data()

    # Load cost config
    cost_config = load_cost_config()
    print(f"Loaded cost configuration for {len(cost_config)} models")

    # Initialize extractor
    extractor = EnhancedAnswerExtractor()

    # Process each entry
    updated_count = 0
    skipped_count = 0

    for entry in tqdm(data, desc="Evaluating entries"):
        # Skip if evaluation_result already exists
        if entry.get('evaluation_result') is not None:
            skipped_count += 1
            continue

        # Evaluate entry
        evaluation_result = evaluate_entry(entry, ground_truth_data, cost_config, extractor)

        if evaluation_result:
            entry['evaluation_result'] = evaluation_result
            updated_count += 1

    # Save updated results
    print(f"\nUpdated {updated_count} entries, skipped {skipped_count} entries")
    save_jsonl(file_path, data)
    print(f"{'='*60}\n")


def main():
    """Main function."""
    # Define files to process
    cached_results_dir = os.path.join(project_root, 'cached_results')
    files_to_process = [
        os.path.join(cached_results_dir, 'deepseek-chat.jsonl'),
        os.path.join(cached_results_dir, 'deepseek-reasoner.jsonl')
    ]

    # Check files exist
    for file_path in files_to_process:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

    # Process each file
    for file_path in files_to_process:
        fill_evaluation_results(file_path)

    print("All files processed successfully!")


if __name__ == '__main__':
    main()
