import argparse
from datasets import Dataset
import pandas as pd
import torch
from torch import nn
from modelscope import snapshot_download, AutoTokenizer
import bitsandbytes as bnb
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.huggingface import SwanLabCallback


# 配置参数
def configuration_parameter():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for model")
    # 模型路径相关参数
    parser.add_argument("--model_name", type=str, default="XGenerationLab/XiYanSQL-QwenCoder-7B-2502",
                        help="Path to the model directory downloaded locally")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save the fine-tuned model and checkpoints")
    # 数据集路径
    parser.add_argument("--train_file", type=str, default="./data/train_data.jsonl",
                        help="Path to the training data file in JSONL format")
    # 训练超参数
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length for the input")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass")

    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Number of steps between logging metrics")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Number of steps between saving checkpoints")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for the optimizer")
    # LoRA 特定参数
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA")
    # 额外优化和硬件相关参数
    parser.add_argument("--fp16", type=bool, default=False,
                        help="Use mixed precision (FP16) training")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--train_mode", type=str, default="lora",
                        help="lora or qlora")
    parser.add_argument("--report_to", type=str, default=None,
                        help="Reporting tool for logging (e.g., tensorboard)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer to use during training")

    args = parser.parse_args()
    return args


# 处理数据
def process_data(data: dict, tokenizer, max_seq_length):
    # 处理数据
    input_ids, attention_mask, labels = [], [], []

    system_text = data['system'].strip()
    user_text = data["user"].strip()
    assistant_text = data["assistant"].strip()

    input_text = "system_text:" + system_text + "\n\nUser:" + user_text + "\n\nAssistant:"

    input_tokenizer = tokenizer(
        input_text,
        add_special_tokens=False,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    output_tokenizer = tokenizer(
        assistant_text,
        add_special_tokens=False,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids += (
            input_tokenizer["input_ids"] + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
    )
    attention_mask += input_tokenizer["attention_mask"] + output_tokenizer["attention_mask"] + [1]
    labels += ([-100] * len(input_tokenizer["input_ids"]) + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
               )

    if len(input_ids) > max_seq_length:  # 做一个截断
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names


# 加载模型
def load_model(args, train_dataset, data_collator, model_path):
    # 加载模型
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if args.fp16 else torch.bfloat16,
        "use_cache": False if args.gradient_checkpointing else True,
        "device_map": "auto"
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    # 用于确保模型的词嵌入层参与训练
    model.enable_input_require_grads()
    # 哪些模块需要注入Lora参数
    target_modules = find_all_linear_names(model, args.train_mode)
    # lora参数设置
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    # 应用 PEFT 配置到模型
    model = get_peft_model(model, config)  # 确保传递的是原始模型

    use_bfloat16 = torch.cuda.is_bf16_supported()  # 检查设备是否支持 bf16
    # 配置训练参数
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        seed=args.seed,
        optim=args.optim,
        fp16=args.fp16,
        bf16=not args.fp16 and use_bfloat16,
    )

    swanlab_callback = SwanLabCallback(
        project="XiYanSQL-QwenCoder-finetune",
        experiment_name="XiYanSQL-QwenCoder-7B-lora",
        description="使用析言SQL-通义千问XiYanSQL-QwenCoder-7B-2502模型在Spider数据集上微调，实现test2sql任务。",
        config={
            "model": args.model_name,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "dataset": "Spider数据集",
        },
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
    )
    return trainer


def main():
    args = configuration_parameter()
    print("*****************加载分词器*************************")
    # 加载分词器
    # 在modelscope上下载模型到本地目录下
    model_path = snapshot_download(args.model_name, cache_dir="./model", revision="master")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    print("*****************处理数据*************************")
    # 处理数据
    data = pd.read_json(args.train_file, lines=True)
    train_ds = Dataset.from_pandas(data)
    train_dataset = train_ds.map(process_data,
                                 fn_kwargs={"tokenizer": tokenizer, "max_seq_length": args.max_seq_length},
                                 remove_columns=train_ds.column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")
    # 加载模型
    print("*****************训练*************************")
    trainer = load_model(args, train_dataset, data_collator, model_path)
    # 训练
    trainer.train()
    # 保存lora权重
    trainer.save_model(output_dir='./save-model/XiYanSQL-QwenCoder-7B-lora')


if __name__ == '__main__':
    main()
