import os
import json
import gc
import random
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
from openai import OpenAI
from feedback_prompts import (
    SYSTEM_SUMMARY_PROMPT,
    USER_SUMMARY_PROMPT,
    SYSTEM_FEEDBACK_PROMPT,
    USER_FEEDBACK_TEMPLATE
)

# === Load config ===
with open("inference/config_inference.json") as f:
    CONFIG = json.load(f)

MODEL_ID = CONFIG["model_paths"][CONFIG["model_to_use"]]
OPENAI_KEY = os.getenv(CONFIG["openai_api_key_env_var"])

STUDENT_DATA_DIR = CONFIG["student_data_base"]
OUTPUT_DIR = os.path.join(CONFIG["output_base"])

QUESTION_ID = CONFIG["question_id"] # we graded one question at a time

client = OpenAI(api_key=OPENAI_KEY)
accelerator = Accelerator()


def load_student_data():
    with open(os.path.join(STUDENT_DATA_DIR, f"{QUESTION_ID}.json")) as f:
        return json.load(f)


def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def summarize_with_gpt4(inner):
    """
    We used GPT-4o as the noticing step for feedback during inference. 
    The task here was to summarize if the student got the answer correct or not, and why.
    Then the DPO tuned model would do the "response" step by crafting feedback based on that summary.
    """
    messages = [
        {"role": "system", "content": SYSTEM_SUMMARY_PROMPT},
        {"role": "user", "content": USER_SUMMARY_PROMPT.format(inner=inner)}
    ]
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("[!] GPT-4 Error:", e)
        return None


def run_inference(model, tokenizer, inner, summary):
    """
    Run inference using the DPO model to generate feedback.
    This is the "response" step where the model crafts feedback based on the "noticing" step summary.
    """
    messages = [
        {"role": "system", "content": SYSTEM_FEEDBACK_PROMPT},
        {"role": "user", "content": USER_FEEDBACK_TEMPLATE.format(inner=inner, summary=summary)}
    ]

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            # Below are placeholders that you should adjust depending on your task requirements.
            max_new_tokens=60, # Adjust as needed for feedback length. 
            temperature=0.7, # Control the output diversity. 
            # top_k=50, # Optional: Top-k filtering 
            # top_p=0.9, # Optional: Nucleus sampling 
            pad_token_id=pad_token_id
        )

    generated_tokens = outputs[0, len(inputs["input_ids"][0]):]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def parse_feedback(text):
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        return json.loads(text[start:end])["feedback"]
    except:
        print("[!] Failed to parse feedback from model output.")
        return None


def process_student(model, tokenizer, student_id, prompt, failed_ids):
    summary = summarize_with_gpt4(prompt)
    if not summary:
        failed_ids.append(student_id)
        return

    model_response = run_inference(model, tokenizer, prompt, summary)
    feedback = parse_feedback(model_response)
    if not feedback:
        failed_ids.append(student_id)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{student_id}.json"), "w") as f:
        json.dump({
            "prompt": prompt,
            "summary": summary,
            "decoded_text": model_response,
            "parsed_text": feedback
        }, f)


def main():
    print("Using model:", MODEL_ID)
    print("Output directory:", OUTPUT_DIR)
    input("[Enter to confirm and begin]...")

    torch.cuda.empty_cache()
    gc.collect()

    model, tokenizer = get_model_and_tokenizer(MODEL_ID)
    model = accelerator.prepare(model)

    student_data = load_student_data()
    all_ids = list(student_data.keys())
    random.shuffle(all_ids)

    unprocessed = [uid for uid in all_ids if not os.path.exists(os.path.join(OUTPUT_DIR, f"{uid}.json"))]
    failed = []

    for uid in tqdm(unprocessed, total=len(unprocessed)):
        process_student(model, tokenizer, uid, student_data[uid], failed)

    if failed:
        print("\n[!] Failed student IDs:", failed)


if __name__ == "__main__":
    main()
