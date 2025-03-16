import torch
from pytorch_lightning import *
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset import Celeb
from model import ImageCompression
from model import batch_size
from new_dataset import NewDataset

train_data = NewDataset(train=True)
test_data = NewDataset(train=False)
train_dataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=20)
test_dataLoader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=20)

print('finished loading dataset')
model = ImageCompression()

# checkpoint_last = ModelCheckpoint()
# checkpoint_mse = ModelCheckpoint(monitor="test mse loss")
# checkpoint_ssim = ModelCheckpoint(monitor="test ssim loss")
logger = TensorBoardLogger("fp16")
trainer = Trainer(max_epochs=200, accelerator="gpu", logger=logger)
# Perform learning rate finder
# trainer.tune(model, train_dataloaders=train_dataLoader, val_dataloaders=test_dataLoader)
# trainer.fit(model, train_dataLoader, test_dataLoader,
#             ckpt_path="lightning_logs/version_2/checkpoints/epoch=5-step=13866.ckpt")
trainer.fit(model, train_dataLoader, test_dataLoader, ckpt_path="fp16/lightning_logs/version_2/checkpoints/epoch=42-step=36808.ckpt")