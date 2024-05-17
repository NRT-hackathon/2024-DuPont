import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_identifier = "NousResearch/Llama-2-7b-chat-hf"
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)

# Load the tokenizer from the identifier, not the model object
tokenizer = AutoTokenizer.from_pretrained(base_model_identifier, trust_remote_code=True)

# Load the base model with quantization config
model = AutoModelForCausalLM.from_pretrained(
    base_model_identifier,  # Use the identifier directly
    quantization_config=bnb_config,  # Apply quantization as needed
    device_map="auto",  # Automatic device mapping
    trust_remote_code=True  # Assuming you trust the code you're loading
)

# Path to the fine-tuned model checkpoint
ft_model_path = "/lustre/ea-nrtmidas/users/3205/finetuning_llama2/test_llama/results/checkpoint-500/"

# Load your fine-tuned model, assuming it's compatible with `AutoModelForCausalLM`
ft_model = AutoModelForCausalLM.from_pretrained(ft_model_path).to("cuda")

# Example input for evaluation
eval_prompt = "I have a molecule with this smiles notation C#Cc1[nH]ccc1c1csc2-c3c(C(=O)c12)ccs3. Suggest modifications to increase its highest occupied molecular orbital (homo) -5.17017 value.",

# Tokenize input and prepare it for model input
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# Model evaluation
ft_model.eval()
with torch.no_grad():
    generated_ids = ft_model.generate(**model_input, max_new_tokens=600)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(generated_text)
