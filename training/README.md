# Training Setup

This folder contains code for training a preference-based model using **Direct Preference Optimization (DPO)**. Our approach fine-tunes a base LLM using comparisons between "chosen" and "rejected" feedback examples from teachers.

---

## 📦 Contents

- `train_dpo.py` — Core training script for DPO fine-tuning.
- `config_training.json` — Optional config file with paths, flags, and hyperparameters.
- `requirements.txt` — Python dependencies for both training and inference.
- `compute_requirements.md` — Notes on compute used (GPU, time, memory, etc.).

---

## 🧠 Data Format

Your training dataset should be setup like this to work with the DPOTrainer:
{
  "prompt": ["prompt1", ...., "promptn"],
  "chosen": ["chosen_response_for_prompt1", ..., "chosen_response_for_promptn"],
  "rejected": ["rejected_response_for_prompt1", ..., "rejected_response_for_promptn"],
}
