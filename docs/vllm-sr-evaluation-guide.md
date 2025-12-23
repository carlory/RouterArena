# vLLM-SR RouterArena Evaluation Guide

This document describes how to evaluate vllm-project/semantic-router using RouterArena.

## Prerequisites

### 1. Start the vllm-sr Classification Service

```bash
gh repo clone vllm-project/semantic-router
make download-models
make run-router
```

### 2. Verify Service is Running

```bash
curl -s http://localhost:8080/api/v1/classify/intent \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "What is 2+2?"}'
```

Expected output:

```json
{"classification":{"category":"math","confidence":0.90...},...}
```

---

## Phase 1: Robustness Evaluation (Free, No LLM API Required)

### Step 1.1: Generate sub_10 Baseline Prediction File

> Robustness evaluation requires sub_10/full predictions as a baseline for comparison

```bash
gh repo clone RouteWorks/RouterArena
cd RouterArena
uv sync
uv run python ./router_inference/generate_prediction_file.py vllm-sr sub_10
```

Output file: `router_inference/predictions/vllm-sr.json`

### Step 1.2: Validate sub_10 Prediction File

```bash
uv run python ./router_inference/check_config_prediction_files.py vllm-sr sub_10
```

Expected: Should display 809 entries

### Step 1.3: Generate Robustness Prediction File

```bash
uv run python ./router_inference/generate_prediction_file.py vllm-sr robustness
```

Output file: `router_inference/predictions/vllm-sr-robustness.json`

### Step 1.4: Validate Robustness Prediction File

```bash
uv run python ./router_inference/check_config_prediction_files.py vllm-sr robustness
```

Expected: Should display 420 entries

### Step 1.5: Run Robustness Evaluation

```bash
uv run python ./llm_evaluation/run.py vllm-sr robustness
```

Output: `metrics.json` containing `robustness_score`

### Step 1.6: Check Results

```bash
cat metrics.json
```

**Evaluation Criteria:**

- `robustness_score >= 0.7`: Ready to proceed to Phase 2
- `robustness_score >= 0.8`: Recommended target
- `robustness_score < 0.5`: Router optimization needed

---

## Phase 2: sub_10 Quality Evaluation (Requires LLM API)

> **Note**: This phase will call OpenAI/Claude/Gemini APIs and incur costs

### Prerequisite: Configure API Keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export MISTRAL_API_KEY="..."
```

or use `.env` file

### Step 2.1: Run sub_10 Evaluation

```bash
uv run python ./llm_evaluation/run.py vllm-sr sub_10
```

This step will:

1. Call various LLMs to generate responses
2. Calculate accuracy
3. Calculate cost
4. Calculate RouterArena score
5. Calculate optimality metrics

### Step 2.2: View Results

```bash
cat metrics.json
```

Key metrics:

- `accuracy`: Correctness rate
- `cost`: Total cost
- `ra_score`: RouterArena composite score
- `opt_sel`: Optimal model selection rate

---

## Phase 3: Full Evaluation (Optional, High Cost)

### Step 3.1: Generate Full Prediction File

```bash
uv run python ./router_inference/generate_prediction_file.py vllm-sr full
```

Prediction entries: 8400 items

### Step 3.2: Run Full Evaluation

```bash
uv run python ./llm_evaluation/run.py vllm-sr full
```

> ⚠️ This will incur significant API call costs

---

## Quick Reference

| Phase                 | Command                            | LLM API Required |
| --------------------- | ---------------------------------- | ---------------- |
| Generate Predictions  | `generate_prediction_file.py`      | ❌               |
| Validate Files        | `check_config_prediction_files.py` | ❌               |
| Robustness Evaluation | `run.py ... robustness`            | ❌               |
| sub_10 Evaluation     | `run.py ... sub_10`                | ✅               |
| Full Evaluation       | `run.py ... full`                  | ✅               |

## File Paths

| File                                                   | Purpose                    |
| ------------------------------------------------------ | -------------------------- |
| `router_inference/config/vllm-sr.json`                 | Router config (model pool) |
| `router_inference/router/vllm_sr.py`                   | Router implementation      |
| `router_inference/predictions/vllm-sr.json`            | sub_10/full predictions    |
| `router_inference/predictions/vllm-sr-robustness.json` | Robustness predictions     |
| `metrics.json`                                         | Evaluation results         |

## Additional Notes

### Analyze robustness flips

```bash
uv run python scripts/analyze_robustness_flips.py

warning: The `tool.uv.dev-dependencies` field (used in `pyproject.toml`) is deprecated and will be removed in a future release; use `dependency-groups.dev` instead
Total overlapping entries: 420

============================================================
ROBUSTNESS ANALYSIS
============================================================
Stable: 368 (87.6%)
Flipped: 52 (12.4%)
Robustness Score: 0.8762

============================================================
MODEL STABILITY (per original model)
============================================================
claude-3-haiku-20240307       :   6 stable,   1 flipped (85.7% stable)
gemini-2.0-flash-001          : 334 stable,  43 flipped (88.6% stable)
gpt-4o-mini                   :  28 stable,   8 flipped (77.8% stable)

============================================================
TOP FLIP TRANSITIONS (from -> to)
============================================================
gemini-2.0-flash-001           -> claude-3-haiku-20240307       :  22
gemini-2.0-flash-001           -> gpt-4o-mini                   :  21
gpt-4o-mini                    -> claude-3-haiku-20240307       :   6
gpt-4o-mini                    -> gemini-2.0-flash-001          :   2
claude-3-haiku-20240307        -> gemini-2.0-flash-001          :   1

============================================================
FLIP EXAMPLES
============================================================

gemini-2.0-flash-001 -> claude-3-haiku-20240307:
  [ArcMMLU_98] Please read the following multiple-choice questions and provide the most likely ...
  [Ethics_commonsense_51] Please read the following multiple-choice questions and determine whether the ac...

gemini-2.0-flash-001 -> gpt-4o-mini:
  [Ethics_justice_45] Please read the following multiple-choice questions and determine whether the ac...
  [GeoBench_1002] Please read the following multiple-choice questions and provide the most likely ...

gpt-4o-mini -> claude-3-haiku-20240307:
  [MMLUPro_biology_2980] Please read the following multiple-choice questions and provide the most likely ...
  [MMLUPro_health_4885] Please read the following multiple-choice questions and provide the most likely ...

gpt-4o-mini -> gemini-2.0-flash-001:
  [MMLUPro_history_4810] Please read the following multiple-choice questions and provide the most likely ...
  [MedMCQA_145] Please read the following multiple-choice questions and provide the most likely ...

claude-3-haiku-20240307 -> gemini-2.0-flash-001:
  [QANTA_Science_1360] Please read the following question and provide the correct answer.
```

### Generate Category-to-Model Mapping

Generate optimal category-to-model mapping based on RouterArena cached results.

```bash
uv run scripts/generate_category_to_model_mapping.py  # read for cached_results
```
