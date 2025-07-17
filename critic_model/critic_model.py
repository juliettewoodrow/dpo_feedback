import json
from openai import OpenAI
from critic_prompts import SYSTEM_PROMPT, SCORING_PROMPT, SCORING_PROMPT_BATCH_INSTRUCTIONS

OPEN_AI_API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key
MODEL = "gpt-4o"  # Specify the model you want to use. We used gpt-4o for this task.
client = OpenAI(api_key=OPEN_AI_API_KEY)


def gpt_call(messages):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=MODEL,
    )
    raw_response = chat_completion.choices[0].message.content
    raw_response = raw_response.replace('```', '').replace('json', '').strip()
    return json.loads(raw_response)


def build_prompt(problem_desc, ta_soln, batch_data):
    """
    Constructs the full user prompt for evaluating feedback in batch.
    """
    prompt_lines = [SCORING_PROMPT.strip()]

    # Add shared context
    prompt_lines.append(SCORING_PROMPT_BATCH_INSTRUCTIONS.strip())
    prompt_lines.append(f"Problem description: {problem_desc}")
    prompt_lines.append(f"TA Solution: {ta_soln}")

    # Add student-specific examples
    for item in batch_data:
        prompt_lines.append(f"\n--- Student Uid {item['student_id']} ---")
        prompt_lines.append(f"Student Solution:\n{item['student_soln']}\n")
        prompt_lines.append(f"Feedback:\n{item['feedback']}\n")

    return "\n".join(prompt_lines)



def call_critic_model(problem_desc, ta_soln, batch_data):
    """
    Sends a batch of examples to the OpenAI model and returns the parsed evaluations.
    Batch data should have the format:
    [
        {
            "student_id": "unique_student_id",
            "student_soln": "The student's solution text",
            "feedback": "The feedback given to the student"
        },
        ...
    ]
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": build_prompt(problem_desc, ta_soln, batch_data)}
    ]

    try:
        response = gpt_call(messages)
        
        # If batching, we expect a list of length == len(batch_data)
        if not isinstance(response, list) or len(response) != len(batch_data):
            raise ValueError(f"GPT response is not a list with the expected length! Expected {len(batch_data)} items, got {len(response)}")
        return response
    
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None
