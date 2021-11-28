from mingpt.model import GPT, GPTConfig
from dataset import TokenDataset
from mingpt.trainer import Trainer, TrainerConfig

VOCAB_SIZE = 50257
BLOCK_SIZE = 128
EPOCHS = 100
BATCH_SIZE = 16

train_dataset = TokenDataset(split='train', block_size=BLOCK_SIZE)
valid_dataset = TokenDataset(split='valid', block_size=BLOCK_SIZE)

mconf = GPTConfig(VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=12, n_head=12, n_embd=768)
model = GPT(mconf)

tconf = TrainerConfig(max_epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=6e-4, lr_decay=True, warmup_tokens=512*20,
                      final_tokens=EPOCHS * len(train_dataset) * BLOCK_SIZE, num_workers=4)

trainer = Trainer(model, train_dataset, valid_dataset, tconf)
trainer.train()


