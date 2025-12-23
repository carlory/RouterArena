#!/usr/bin/env python3
"""Analyze robustness flip patterns to identify which model transitions cause low scores."""

import json
from collections import Counter, defaultdict
from pathlib import Path


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
            
            # Store example
            prompt = full_preds[idx].get("prompt", "")[:80]
            if len(flip_examples[transition]) < 3:
                flip_examples[transition].append((idx, prompt))
    
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
