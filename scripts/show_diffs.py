#!/usr/bin/env python3
import csv

with open('router_inference/robustness_flips.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f'Total rows: {len(rows)}\n')
print('Sample rows with differences:\n')

for i, row in enumerate(rows[:5]):
    print(f'{i+1}. idx={row["idx"]}')
    print(f'   Differences: {row["prompt_differences"]}')
    print(f'   Original: {row["prompt"][:100]}...')
    print(f'   Robust:   {row["robust_prompt"][:100]}...')
    print()
