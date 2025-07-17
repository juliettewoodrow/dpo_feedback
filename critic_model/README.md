## Critic Model

This directory contains a custom evaluation pipeline that uses an LLM to score feedback on three dimensions:

- **Assertiveness**: Is the feedback bold and specific or timid and vague?
- **Accuracy**: Does the feedback reflect the student's actual work correctly?
- **Helpfulness**: Does the feedback offer value or guidance to the student?

### Files

- `critic_model.py`: Contains the logic to call the OpenAI API and evaluate batches of feedback examples.
- `critic_prompts.py`: Defines the system prompt and scoring instructions used for evaluation.

### How to Use

1. Format your batch of feedback examples as a list of dictionaries:
    ```python
    batch_data = [
        {
            "student_id": "s1",
            "student_soln": "...",
            "feedback": "..."
        },
        ...
    ]
    ```

2. Provide a `problem_desc` and `ta_soln` string.

3. Call:
    ```python
    from critic_model import call_critic_model
    results = call_critic_model(problem_desc, ta_soln, batch_data)
    ```

4. The result will be a list of JSON objects with evaluations for each batch.

---

### ⚙️ Why Batching Helps

We found that the critic model evaluates feedback more effectively when it sees multiple examples at once. Batching allows it to calibrate judgments relatively which helps the consistency of scores. Start with ~5 examples per batch, but increase or decrease depending on the length of the student solutions. Each example is scored independently, and the model returns a clean JSON array of results.
