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


base_model = "NousResearch/Llama-2-7b-chat-hf"
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)
# Load the tokenizer from the identifier, not the model object
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config)
eval_prompt = "I have a molecule with this smiles notation C#Cc1[nH]ccc1c1csc2-c3c(C(=O)c12)ccs3. Suggest modifications to increase its highest occupied molecular orbital (homo) -5.17017 value.",
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0],
                           skip_special_tokens=True))
