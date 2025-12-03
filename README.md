# Quantifying Reasoning Followability via Handoff Success Rate

This is a repo for the project **Quantifying Reasoning Followability**.
We introduce **Handoff Success Rate (HSR)**, a measure of the followability of Chain-of-Thought (CoT) reasoning by testing if a weaker "Junior" model can successfully complete a reasoning trace started by a stronger "Senior" model.
Standard interpretability tools struggle with multi-step, stochastic reasoniong.
This project uses **Handoff Robustness** as an analysis lens.
We aim to empirically pinpoint "cognitive leaps" in reasoning traces that are brittle or non-transferable to weaker agents.

## Repo Structure

```
handoff-interp/
├── data/                   # Local storage for datasets and generated traces
│   ├── gsm8k/              # Downloaded/processed GSM8K
│   └── senior_traces/      # Output from Stage A (M_sen traces)
├── results/                # Output from Stage B (Handoff experiments)
│   ├── exp1_sanity/        # Logs from Experiment 1
│   └── figures/            # Generated plots
├── src/                    # Source code
│   ├── __init__.py
│   ├── generation.py       # Logic for M_sen generating full CoT
│   ├── handoff.py          # Logic for M_jun taking over (The HSR Engine)
│   ├── text_utils.py       # Sentence splitting, prompt formatting, cleaning
│   └── analysis.py         # Metrics calculation (HSR, Delta HSR)
├── scripts/                # Shell scripts for reproducibility
│   ├── 01_gen_senior.sh    # Run Stage A
│   └── 02_exp1_sanity.sh   # Run Stage B (Exp 1)
├── notebooks/              # Jupyter notebooks for quick visualization
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

### `src/generation.py`

Generates baseline "Golden Traces" using a Senior Model (e.g., Qwen3-32B).

### `src/handoff.py`

The core engine.
Takes a Golden Trace, truncates it at step $k$, and samples $N$ completions from a Junior model to calculate HSR.

### `src/text_utils.py`

Handles the critical logic of robust sentence splitting and cross-model prompt formatting.

