import os
import json
import torch
import gc
from datetime import datetime

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig, apply_chat_template
from peft import LoraConfig

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def load_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def format_and_tokenize_dpo_data(data_json, tokenizer):
    formatted_data = []

    prompts = data_json["prompt"]
    chosens = data_json["chosen"]
    rejecteds = data_json["rejected"]

    for i in range(len(prompts)):
        prompt = prompts[i]
        chosen = eval(chosens[i])["feedback"]
        rejected = eval(rejecteds[i])["feedback"]

        messages = {
            "prompt": [{"role": "user", "content": prompt}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}]
        }

        if sum(len(m["content"]) for m in messages["prompt"] + messages["chosen"] + messages["rejected"]) < 30000:
            formatted = apply_chat_template(messages, tokenizer)
            if all(len(tokenizer.encode(formatted[key])) < 30000 for key in ["prompt", "chosen", "rejected"]):
                formatted_data.append(formatted)

    return Dataset.from_list(formatted_data)



def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_folder="offload_folder"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer



def setup_dpo_trainer(model, tokenizer, config, train_dataset, eval_dataset, output_dir, timestamp):
    training_args = DPOConfig(
        run_name=f"dpo_instruct_{timestamp}",
        output_dir=output_dir,
        beta=config["beta"],
        warmup_steps=config["warmup_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        fp16=True,
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        save_strategy="steps",
        logging_steps=config["logging_steps"],
        max_prompt_length=config["max_prompt_length"],
        max_length=config["max_length"],
        num_train_epochs=config["num_train_epochs"],
        torch_compile=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        truncation_mode="keep_end"
    )

    lora = config["lora_config"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        bias=lora["bias"],
        task_type=lora["task_type"],
        target_modules=lora["target_modules"]
    )

    return DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config
    )


if __name__ == "__main__":
    config = load_config("config_dpo.json")

    torch.cuda.empty_cache()
    gc.collect()

    base_model = config["sft_model_to_load"] # we first tuned an model with supervised fine-tuning and used that as our base model. You can start with any base model though.
    model, tokenizer = load_model_and_tokenizer(config["sft_model_to_load"])
    raw_data = load_json(config["data_path"])
    dataset = format_and_tokenize_dpo_data(raw_data, tokenizer)

    train_set, eval_set = your_custom_split_function(dataset)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(config["output_dir"], f"dpo_model_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    trainer = setup_dpo_trainer(model, tokenizer, config, train_set, eval_set, output_dir, timestamp)
    print("Starting training...")
    trainer.train()
    print("Training complete!")

    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
