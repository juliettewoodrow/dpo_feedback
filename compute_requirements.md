# Compute Resources and Infrastructure

This file details the computational resources and infrastructure used to train and evaluate models in our DPO feedback pipeline.

## Model Training Setup

- **Base model**: [Meta Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- **Trainer**: Hugging Face's `DPOTrainer`
- **Training duration per run**: ~7 hours
- **GPUs used**: 3 × RTX A6000 (48–49GB each)
- **Average training set size**: ~1,408 examples (preferences)
- **Training schedule**: One DPO training job per assignment release (i.e., once per week)

## Inference Setup

- **Inference GPU usage**: 1 × RTX A6000
- To run inference to generate feedback for a class size of 300 students for 1 problem took around ~1 hour with this setup. 

