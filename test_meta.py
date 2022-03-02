import torch
from models import ShardedGPT, GPTSmallConfig



vocab_size = 3072
block_size = 128
gpt_small = ShardedGPT(config=GPTSmallConfig(vocab_size=vocab_size, block_size=block_size))

print("initialized gpt small")
