from __future__ import annotations
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    _ = parser.parse_args()

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    data_path = "data/processed/datasets/sft/train.jsonl"
    output_dir = "training/checkpoints/qwen25_7b_sft_expdesign"

    dataset = load_dataset("json", data_files={"train": data_path})["train"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        max_seq_length=3072,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=200,
        bf16=True,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
