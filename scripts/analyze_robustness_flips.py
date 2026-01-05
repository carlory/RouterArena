#!/usr/bin/env python3
"""Analyze robustness flip patterns to identify which model transitions cause low scores."""

import csv
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Tuple


API_URL = "http://localhost:8080/api/v1/classify/intent"


def classify_query(query: str) -> Optional[Tuple[str, float]]:
    """Call vllm-sr API to classify a query into a category with confidence.

    Returns:
        Tuple of (category, confidence) or None if classification fails
    """
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
            classification = result.get("classification", {})
            category = classification.get("category", "other")
            confidence = classification.get("confidence", 0.0)
            return (category, confidence)
    except Exception as e:
        print(f"Classification failed: {e}")
        return None


def load_predictions(path: str) -> dict:
    """Load predictions and index by global_index."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        str(entry.get("global index") or entry.get("global_index")): entry
        for entry in data
        if not entry.get("for_optimality", False)
    }


def main():
    base_path = Path("router_inference/predictions")
    
    # Load both prediction files
    full_path = base_path / "vllm-sr.json"
    robust_path = base_path / "vllm-sr-robustness.json"
    
    if not full_path.exists() or not robust_path.exists():
        print(f"Missing files: {full_path} or {robust_path}")
        return
    
    full_preds = load_predictions(str(full_path))
    robust_preds = load_predictions(str(robust_path))
    
    # Find overlapping indices
    common_indices = set(full_preds.keys()) & set(robust_preds.keys())
    print(f"Total overlapping entries: {len(common_indices)}")
    
    # Analyze flips
    flip_transitions = Counter()  # (from_model, to_model) -> count
    flip_examples = defaultdict(list)  # (from_model, to_model) -> [(idx, prompt_snippet)]
    stable_count = 0
    flip_count = 0

    model_flip_stats = defaultdict(lambda: {"stable": 0, "flipped": 0})

    # CSV data for flips
    flip_data = []

    for idx in sorted(common_indices):
        full_model = full_preds[idx].get("prediction", "")
        robust_model = robust_preds[idx].get("prediction", "")

        if full_model == robust_model:
            stable_count += 1
            model_flip_stats[full_model]["stable"] += 1
        else:
            flip_count += 1
            transition = (full_model, robust_model)
            flip_transitions[transition] += 1
            model_flip_stats[full_model]["flipped"] += 1

            # Get prompts
            full_prompt = full_preds[idx].get("prompt", "")
            robust_prompt = robust_preds[idx].get("prompt", "")

            # Store example snippet
            prompt_snippet = full_prompt[:80]
            if len(flip_examples[transition]) < 3:
                flip_examples[transition].append((idx, prompt_snippet))

            # Classify queries for flips
            print(f"Classifying flip {flip_count}/{len(common_indices)}: idx={idx}")

            full_classification = classify_query(full_prompt)
            robust_classification = classify_query(robust_prompt)

            if full_classification and robust_classification:
                full_category, full_confidence = full_classification
                robust_category, robust_confidence = robust_classification

                flip_data.append({
                    "idx": idx,
                    "prompt": full_prompt,
                    "full_model": full_model,
                    "full_category": full_category,
                    "full_confidence": full_confidence,
                    "robust_prompt": robust_prompt,
                    "robust_model": robust_model,
                    "robust_category": robust_category,
                    "robust_confidence": robust_confidence,
                    "prompt_differences": "Pending analysis",
                })

    # Save flips to CSV
    if flip_data:
        csv_path = base_path.parent / "robustness_flips.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "idx",
                "prompt",
                "full_model",
                "full_category",
                "full_confidence",
                "robust_prompt",
                "robust_model",
                "robust_category",
                "robust_confidence",
                "prompt_differences"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flip_data)
        print(f"\nFlip data saved to: {csv_path}")

    print(f"\n{'='*60}")
    print(f"ROBUSTNESS ANALYSIS")
    print(f"{'='*60}")
    print(f"Stable: {stable_count} ({stable_count/len(common_indices)*100:.1f}%)")
    print(f"Flipped: {flip_count} ({flip_count/len(common_indices)*100:.1f}%)")
    print(f"Robustness Score: {1 - flip_count/len(common_indices):.4f}")
    
    print(f"\n{'='*60}")
    print(f"MODEL STABILITY (per original model)")
    print(f"{'='*60}")
    for model in sorted(model_flip_stats.keys()):
        stats = model_flip_stats[model]
        total = stats["stable"] + stats["flipped"]
        stability = stats["stable"] / total * 100 if total > 0 else 0
        print(f"{model:30s}: {stats['stable']:3d} stable, {stats['flipped']:3d} flipped ({stability:.1f}% stable)")
    
    print(f"\n{'='*60}")
    print(f"TOP FLIP TRANSITIONS (from -> to)")
    print(f"{'='*60}")
    for (from_model, to_model), count in flip_transitions.most_common(15):
        print(f"{from_model:30s} -> {to_model:30s}: {count:3d}")
    
    print(f"\n{'='*60}")
    print(f"FLIP EXAMPLES")
    print(f"{'='*60}")
    for (from_model, to_model), examples in list(flip_examples.items())[:5]:
        print(f"\n{from_model} -> {to_model}:")
        for idx, prompt in examples[:2]:
            print(f"  [{idx}] {prompt}...")

if __name__ == "__main__":
    main()
