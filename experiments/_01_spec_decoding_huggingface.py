#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False
)

#%%
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B-Instruct")
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-13B-Instruct", quantization_config=quantization_config, device_map="auto", torch_dtype=torch.bfloat16)
# %%
model.eval()
model