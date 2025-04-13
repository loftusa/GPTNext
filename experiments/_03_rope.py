#%%
import torch
import torch.nn as nn
from _03_rope_model import GPT, GPTConfig

gpt_large = GPT.from_pretrained(model_type='gpt2-xl')
gpt2 = GPT.from_pretrained(model_type='gpt2')