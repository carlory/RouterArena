#!/usr/bin/env python3
"""
Generate optimal category-to-model mapping based on RouterArena cached results.

This script:
1. Loads cached LLM results from cached_results/*.jsonl
2. Classifies each query using vllm-sr API
3. Computes average score per (category, model) pair
4. Outputs the optimal model for each category
"""

import json
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


API_URL = "http://localhost:8080/api/v1/classify/intent"
CACHED_RESULTS_DIR = Path("cached_results")
DATASET_PATH = Path("dataset/router_data_10.json")


def load_cached_results() -> Dict[str, Dict[str, Dict]]:
    """
    Load all cached results indexed by global_index and model.
    Returns: {global_index: {model_name: {score, cost, ...}}}
    """
    results: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    
    for jsonl_file in CACHED_RESULTS_DIR.glob("*.jsonl"):
        model_name = jsonl_file.stem  # e.g., "gpt-4o-mini"
        print(f"Loading {jsonl_file}...")
        
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    global_index = entry.get("global_index")
                    if global_index:
                        eval_result = entry.get("evaluation_result", {})
                        results[global_index][model_name] = {
                            "score": eval_result.get("score", 0.0),
                            "cost": eval_result.get("inference_cost", 0.0),
                        }
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(results)} unique queries")
    return results


def classify_query(query: str) -> Optional[str]:
    """Call vllm-sr API to classify a query into a category."""
    try:
        data = json.dumps({"text": query}).encode("utf-8")
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("classification", {}).get("category", "other")
    except Exception as e:
        return None


def load_dataset() -> List[Dict]:
    """Load the dataset with queries."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("OPTIMAL CATEGORY-MODEL MAPPING GENERATOR")
    print("=" * 60)
    
    # Step 1: Load cached results
    cached = load_cached_results()
    models = set()
    for model_scores in cached.values():
        models.update(model_scores.keys())
    print(f"Models found: {sorted(models)}")
    
    # Step 2: Load dataset
    dataset = load_dataset()
    print(f"Dataset entries: {len(dataset)}")
    
    # Step 3: Classify each query and collect scores
    # Structure: {category: {model: [scores]}}
    category_model_scores: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    category_model_costs: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    
    classified_count = 0
    failed_count = 0
    
    print("\nClassifying queries...")
    for i, entry in enumerate(dataset):
        global_index = entry.get("global index") or entry.get("global_index")
        prompt = entry.get("prompt_formatted", "")
        
        if not global_index or global_index not in cached:
            continue
        
        # Classify
        category = classify_query(prompt)
        if not category:
            failed_count += 1
            continue
        
        classified_count += 1
        
        # Record scores for each model
        for model, data in cached[global_index].items():
            category_model_scores[category][model].append(data["score"])
            category_model_costs[category][model].append(data["cost"])
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} queries...")
    
    print(f"\nClassified: {classified_count}, Failed: {failed_count}")
    
    # Step 4: Compute averages and find optimal
    print("\n" + "=" * 60)
    print("CATEGORY ANALYSIS")
    print("=" * 60)
    
    optimal_mapping: Dict[str, str] = {}
    
    for category in sorted(category_model_scores.keys()):
        print(f"\n### {category.upper()} ###")
        print(f"{'Model':<30s} {'Avg Score':>10s} {'Avg Cost':>12s} {'Count':>8s}")
        print("-" * 60)
        
        best_model = None
        best_score = -1
        
        for model in sorted(category_model_scores[category].keys()):
            scores = category_model_scores[category][model]
            costs = category_model_costs[category][model]
            avg_score = sum(scores) / len(scores) if scores else 0
            avg_cost = sum(costs) / len(costs) if costs else 0
            
            print(f"{model:<30s} {avg_score:>10.4f} ${avg_cost:>11.6f} {len(scores):>8d}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        if best_model:
            optimal_mapping[category] = best_model
            print(f"  → Best: {best_model} (score: {best_score:.4f})")
    
    # Step 5: Output the mapping
    print("\n" + "=" * 60)
    print("OPTIMAL MAPPING (for vllm_sr.py)")
    print("=" * 60)
    print("\nCATEGORY_MODEL_MAPPING: Dict[str, str] = {")
    for category, model in sorted(optimal_mapping.items()):
        print(f'    "{category}": "{model}",')
    print("}")
    
    # Step 6: Save to file
    output_path = Path("scripts/optimal_mapping.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(optimal_mapping, f, indent=2)
    print(f"\nMapping saved to {output_path}")


if __name__ == "__main__":
    main()
