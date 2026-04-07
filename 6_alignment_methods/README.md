# 6 Alignment Methods (Step by Step)

This folder contains a practical, runnable progression from basic preference data concepts to DPO [Direct preference optimization ]and RLHF [Reinforcement Learning based Human feedback] intuition.

## Learning order

1. `step1_preference_dataset_basics.py`
2. `step2_dpo_intuition.py`
3. `step3_rlhf_intuition.py`
4. `step4_preference_dataset_pipeline.py`
5. `step5_ollama_preference_judge.py`

## What each step teaches

### Step 1: Preference Dataset Basics
- How to structure `prompt / chosen / rejected`
- Why agreement and label consistency matter
- How to split train/validation and inspect quality slices

### Step 2: DPO Intuition
- Direct Preference Optimization objective intuition
- Why chosen should become more likely than rejected
- How reference anchoring controls policy drift

### Step 3: RLHF Intuition
- Reward model training from pairwise preferences (Bradley-Terry)
- Policy update with a PPO-style clipped objective (toy)
- Tradeoff between reward gain and KL stability

### Step 4: Preference Dataset Pipeline
- Loads preference JSONL and validates schema
- Cleans, deduplicates, and splits train/validation
- Writes processed files plus quality report

### Step 5: Ollama Preference Judge
- Generates two candidate outputs per prompt (Model A vs Model B)
- Uses a judge model to pick winner and reason
- Saves prompt/chosen/rejected JSONL for DPO-style training

## Run

From workspace root:

```powershell
python 6_alignment_methods/step1_preference_dataset_basics.py
python 6_alignment_methods/step2_dpo_intuition.py
python 6_alignment_methods/step3_rlhf_intuition.py

# Step 4: generate sample input then build processed dataset
python 6_alignment_methods/step4_preference_dataset_pipeline.py --make-sample
python 6_alignment_methods/step4_preference_dataset_pipeline.py

# Step 5: collect preference pairs from Ollama
python 6_alignment_methods/step5_ollama_preference_judge.py --max-pairs 4
```

## Notes

- These are educational toy implementations for intuition building.
- They are intentionally lightweight and dependency-minimal.
- Use this as conceptual preparation before applying full frameworks (TRL, Axolotl, etc.).
- Step 5 requires local Ollama running with at least one pulled model.
