import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset import Celeb
from model import ImageCompression
from model import batch_size

train_data = Celeb(train=True)
test_data = Celeb(train=False)
train_dataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
test_dataLoader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

print('finished loading dataset')
# model = ImageCompression()
# checkpoint_callback = ModelCheckpoint(dirpath="lightning_logs/version_13")
# model = ImageCompression.load_from_checkpoint(checkpoint_path="lightning_logs/version_24_new_start_from_40/checkpoints/epoch=28-step=51793.ckpt")
model = ImageCompression()

checkpoint_last = ModelCheckpoint()
checkpoint_mse = ModelCheckpoint(monitor="test mse loss")
checkpoint_ssim = ModelCheckpoint(monitor="test ssim loss")

trainer = Trainer(gpus=1, max_epochs=200, callbacks=[checkpoint_last, checkpoint_mse, checkpoint_ssim],
                  resume_from_checkpoint="lightning_logs/version_0/checkpoints/epoch=36-step=66081.ckpt")
# trainer = Trainer(gpus=1, precision=16)
trainer.fit(model, train_dataLoader, test_dataLoader)
