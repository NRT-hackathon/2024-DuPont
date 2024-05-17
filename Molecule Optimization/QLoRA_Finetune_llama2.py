# import libraries
import os
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline,logging, )
from peft import LoraConfig
from trl import SFTTrainer
import matplotlib.pyplot as plt
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import transformers
import wandb
from datetime import datetime


# Accelarator
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

wandb.login(key='')
wandb_project = "sri-llama-2-7b-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

# base model
base_model = "NousResearch/Llama-2-7b-chat-hf"
new_model = "llama-2-7b-chat-sri"

# custom dataset
sri_dataset = "/lustre/ea-nrtmidas/users/3205/finetuning_llama3/test_llama/prompt_dataset.jsonl"
dataset = load_dataset('json', data_files=sri_dataset, split='train')
train_dataset, eval_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()


def formatting_func(example):
    text = f"### Instruction: {example['instruction']}\n ### Question: {example['input']}\n ### Answer: {example['output']}"
    return text


compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)

model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", add_eos_token=True, add_bos_token=True,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)


def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset, save_dir):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Create the directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')

    # Save the plot to a file
    file_path = os.path.join(save_dir, 'input_ids_length_distribution.png')
    plt.savefig(file_path)
    plt.close()
    print(f"Plot saved to {file_path}")


save_directory = '/lustre/ea-nrtmidas/users/3205/finetuning_llama3/test_llama/'
plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset, save_directory)

max_length = 512  # This was an appropriate max length for my dataset


def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)
print(tokenized_train_dataset[1]['input_ids'])
plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset, save_directory)
eval_prompt = "Analyze the change in the gap between HOMO and LUMO for C#Cc1[nH]ccc1c1csc2-c3c(C(=O)c12)ccs3 and its similar molecule CC#Cc1[nH]ccc1c1csc2-c3c(C(=O)c12)ccs3."
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0],
                           skip_special_tokens=True))

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print(model)
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)

if torch.cuda.device_count() > 1:  # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True
run_name = "llama-2-7b-finetune"
tokenizer.pad_token = tokenizer.eos_token
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir="./results",
        warmup_steps=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_steps=500,
        learning_rate=1e-4,  # Want a small lr for finetuning
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",  # Directory for storing logs
        save_strategy="steps",  # Save the model checkpoint every logging step
        save_steps=25,  # Save checkpoints every 25 steps
        evaluation_strategy="steps",  # Evaluate the model every logging step
        eval_steps=50,  # Evaluate and save checkpoints every 50 steps
        do_eval=True,  # Perform evaluation at the end of training
	report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"  
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
print('finished')