#!/usr/bin/env python3
"""Analyze differences between prompt and robust_prompt in the CSV file."""

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path


def analyze_differences(original: str, robust: str) -> str:
    """Analyze and categorize differences between two prompts."""
    if original == robust:
        return "No differences"

    differences = []

    # Split into words
    orig_words = original.split()
    rob_words = robust.split()

    # Count different types of changes
    spelling_errors = 0
    word_deletions = 0
    word_additions = 0
    case_changes = 0

    # Use SequenceMatcher to find differences
    matcher = SequenceMatcher(None, orig_words, rob_words)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Check if it's spelling errors or word substitutions
            for orig_word, rob_word in zip(orig_words[i1:i2], rob_words[j1:j2]):
                orig_clean = re.sub(r'[^\w]', '', orig_word).lower()
                rob_clean = re.sub(r'[^\w]', '', rob_word).lower()

                if orig_clean == rob_clean:
                    # Same word, different case or punctuation
                    if orig_word != rob_word:
                        case_changes += 1
                else:
                    # Different words - likely spelling error if similar
                    ratio = SequenceMatcher(None, orig_clean, rob_clean).ratio()
                    if ratio > 0.5:  # Similar enough to be spelling error
                        spelling_errors += 1
        elif tag == 'delete':
            word_deletions += (i2 - i1)
        elif tag == 'insert':
            word_additions += (j2 - j1)

    # Build result string
    if spelling_errors > 0:
        differences.append(f"Spelling errors ({spelling_errors})")
    if word_deletions > 0:
        differences.append(f"Word deletions ({word_deletions})")
    if word_additions > 0:
        differences.append(f"Word additions ({word_additions})")
    if case_changes > 0:
        differences.append(f"Case/punctuation changes ({case_changes})")

    return "; ".join(differences) if differences else "Unknown differences"


def main():
    csv_path = Path("router_inference/robustness_flips.csv")

    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} rows...")

    # Analyze each row
    for i, row in enumerate(rows, 1):
        prompt = row['prompt']
        robust_prompt = row['robust_prompt']

        diff = analyze_differences(prompt, robust_prompt)
        row['prompt_differences'] = diff

        if i % 10 == 0:
            print(f"Processed {i}/{len(rows)} rows...")

    # Write updated CSV
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nUpdated CSV saved to: {csv_path}")

    # Print summary statistics
    diff_types = {}
    for row in rows:
        diff = row['prompt_differences']
        diff_types[diff] = diff_types.get(diff, 0) + 1

    print("\nDifference type distribution:")
    for diff_type, count in sorted(diff_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {diff_type}: {count}")


if __name__ == "__main__":
    main()
