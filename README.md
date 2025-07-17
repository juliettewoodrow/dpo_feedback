# Improving Generative AI Student Feedback: Direct Preference Optimization with Teachers in the Loop
**Authors**  
Juliette Woodrow, Sanmi Koyejo, Chris Piech  
Stanford University

We present a system for improving LLM-generated student feedback through **Direct Preference Optimization (DPO)**, using real-time preferences from teachers during grading. The system was deployed in two offerings of a Stanford University course on probability and evaluated both through expert blind review and automated critic models. You can read the whole paper here: [Improving Generative AI Student Feedback: Direct Preference Optimization with Teachers in the Loop (Woodrow et al., 2025)](https://juliettewoodrow.github.io/paper-hosting/dpo_feedback.pdf)

You can learn more about Direct Preference Optimization from their paper: [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)

## What's in This Repo

### Model Code

**Training Setup**
- `train_dpo.py`: Code for fine-tuning LLMs with DPO.
- `requirements.txt`: Python package dependencies.
- `compute_requirements.md`: Notes on GPU setup, training time, and dataset sizes.

**Inference Setup**
- `generate_feedback.py`: Script for generating feedback using a fine-tuned model.
- `prompts/`: Structured prompts used during inference.
- `requirements.txt`: Python package dependencies. 

### Evaluation Setup

**Custom Critic Model**
- `critic_model.py`: Code to evaluate feedback on accuracy, helpfulness, and assertiveness.
- `critic_prompts/`: Prompts for our custom LLM-based critic.
- `requirements.txt`: Evaluation setup (uses OpenAI API).

### Additional Analyses

**Fairness Evaluation**
[TODO]
- `counterfactual_outline.md`: Our proposed framework for counterfactual fairness in this space.

**Qualitative Feedback**
- `qualitative_feedback.md`: Anonymous TA justifications from our controlled human study, comparing DPO and GPT-4o.

### How to Get the Code? 
You can find all of the code here: [Github Repo](https://github.com/juliettewoodrow/dpo_feedback)

