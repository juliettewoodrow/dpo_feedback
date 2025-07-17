# Improving Generative AI Student Feedback: Direct Preference Optimization with Teachers in the Loop
**Authors**  
Juliette Woodrow, Sanmi Koyejo, Chris Piech  
Stanford University

We present a system for improving LLM-generated student feedback through **Direct Preference Optimization (DPO)**, using real-time preferences from teachers during grading. The system was deployed in two offerings of a Stanford University course on probability and evaluated both through expert blind review and automated critic models. 

[Read the paper Improving Generative AI Student Feedback: Direct Preference Optimization with Teachers in the Loop (Woodrow et al., 2025)](https://juliettewoodrow.github.io/paper-hosting/dpo_feedback.pdf)  
[View the full GitHub repo](https://github.com/juliettewoodrow/dpo_feedback)  
Learn more about DPO: [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)

## What's in This Repo

### Model Code

**Training Setup**
- `train_dpo.py`: Code for fine-tuning LLMs with DPO.


**Inference Setup**
- `generate_feedback.py`: Script for generating feedback using a fine-tuned model.
- `prompts/`: Structured prompts used during inference.

### Evaluation Setup

**Custom Critic Model**
- `critic_model.py`: Code to evaluate feedback on accuracy, helpfulness, and assertiveness.
- `critic_prompts/`: Prompts for our custom LLM-based critic.

`compute_requirements.md`: Notes on GPU setup, training time, and dataset sizes.
`requirements.txt`: Python package dependencies.

### How to Get the Code? 
You can find all of the code here: [Github Repo](https://github.com/juliettewoodrow/dpo_feedback)

