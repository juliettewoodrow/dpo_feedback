# Improving Generative AI Student Feedback: Direct Preference Optimization with Teachers in the Loop

This repository supports our paper:

📄 [Read the paper (PDF)](https://juliettewoodrow.github.io/paper-hosting/dpo_feedback.pdf)

We present a system for improving LLM-generated student feedback through **Direct Preference Optimization (DPO)**, using real-time preferences from teachers during grading. The system was deployed in two offerings of a Stanford University course on probability and evaluated both through expert blind review and automated critic models.

---

## 💡 What's in This Repo

### 🧠 Model Code

**Training Setup**
- `train_dpo.py`: Code for fine-tuning LLMs with DPO.
- `requirements.txt`: Python package dependencies.
- `compute_requirements.md`: Notes on GPU setup, training time, and dataset sizes.

**Inference Setup**
- `generate_feedback.py`: Script for generating feedback using a fine-tuned model.
- `prompts/`: Structured prompts used during inference.
- `requirements.txt`: Python package dependencies. 

---

### 📊 Evaluation Setup

**Custom Critic Model**
- `critic_model.py`: Code to evaluate feedback on accuracy, helpfulness, and assertiveness.
- `critic_prompts/`: Prompts for our custom LLM-based critic.
- `requirements.txt`: Evaluation setup (uses OpenAI API).

---

### 📈 Additional Analyses

**Fairness Evaluation**
[TODO]
- `counterfactual_outline.md`: Our proposed framework for counterfactual fairness in this space.

**Qualitative Feedback**
- `qualitative_feedback.md`: Anonymous TA justifications from our controlled human study, comparing DPO and GPT-4o.

---

## 🔒 Ethical Considerations

We do **not** release any student submissions or feedback data to preserve privacy. This repository is intended to help instructors **build their own pipelines** using the released methods.

---
