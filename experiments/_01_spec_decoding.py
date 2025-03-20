#%%
import torch
import torch.nn as nn

from _01_spec_decoding_model import GPT, GPTConfig


gpt_large = GPT.from_pretrained(model_type='gpt2-xl')
gpt2 = GPT.from_pretrained(model_type='gpt2')


