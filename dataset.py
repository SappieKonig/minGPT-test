from torch.utils.data import Dataset
import torchtext
from transformers import GPT2Tokenizer
import numpy as np
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

train_data = torchtext.datasets.WikiText2(split='train')
valid_data = torchtext.datasets.WikiText2(split='valid')
test_data = torchtext.datasets.WikiText2(split='valid')


class TokenDataset(Dataset):

    def __init__(self, split, block_size):
        self.data = torchtext.datasets.WikiText2(split=split)
        self.processed_data = np.concatenate([tokenizer(i)['input_ids'] for i in self.data])
        self.block_size = block_size

    def __len__(self):
        return (len(self.processed_data) - 1) // self.block_size

    def __getitem__(self, idx):
        chunk = self.processed_data[idx*self.block_size: (idx+1)*self.block_size+1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])