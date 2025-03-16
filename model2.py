import torch
from torch import nn
from lightning import LightningModule
from torch.optim import Adam, lr_scheduler
from torchjpeg import quantization, dct, metrics
import warnings
from dataset import img_size
import segmentation_models_pytorch as smp

warnings.filterwarnings('ignore')
batch_size = 40
learning_rate = 0.001
ssim_scale = 5

class Jpeg(nn.Module):
    def __init__(self, quality):
        super().__init__()
        q = torch.tensor([[quality]], dtype=torch.float)
        scale = quantization.ijg.qualities_to_scale_factors(q)

        luma = quantization.ijg.scale_quantization_matrices(scale, "luma")
        chroma = quantization.ijg.scale_quantization_matrices(scale, "chroma")

        luma = luma.reshape(8, 8).unsqueeze(0).unsqueeze(0).contiguous()
        chroma = chroma.reshape(8, 8).unsqueeze(0).unsqueeze(0).contiguous()
        self.mat = torch.cat((luma, chroma), dim=1)

    def forward(self, x):
        dctC = dct.images_to_batch(x)
        # import ipdb;ipdb.set_trace()
        if dctC.get_device() >= 0:
            self.mat = self.mat.to(dctC.get_device())
        y, cb, cr = quantization.quantize_multichannel(dctC, self.mat, round_func=Jpeg.diff_round)
        dctC = quantization.dequantize_multichannel(y, cb, cr, mat=self.mat)
        return dct.batch_to_images(dctC) - 0.5



    @staticmethod
    def diff_round(x):
        return torch.round(x) + (x - torch.round(x)) ** 3

class Encoder(nn.Module):
    def __init__(self, architecture ,backbone):
        super().__init__()
        self.model = smp.create_model(
            arch=architecture,
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            upsampling=2,
            classes=3,
            activation="tanh",
        )
    def forward(self, x):
        x = self.model(x)
        return (x+1)/2


class Decoder(nn.Module):
    def __init__(self, architecture, backbone):
        super().__init__()
        self.model = smp.create_model(
            arch=architecture,
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            upsampling=8,
            classes=3,
            activation="tanh",
        )

    def forward(self, x):
        x = self.model(x)
        return x

class ImageCompression2(LightningModule):
    def __init__(self, q=50):
        super().__init__()
        self.Loss = nn.L1Loss(reduction="none")
        self.jpeg = Jpeg(quality=q)
        self.encoder = Encoder(architecture="pan", backbone="mobileone_s4")
        self.decoder = Decoder(architecture="pan", backbone="mobileone_s4")
        self.lr = learning_rate

    def forward(self, img):
        # import ipdb;ipdb.set_trace()
        img = self.encoder(img)
        img = self.jpeg(img)
        if torch.isnan(img).any():
            import ipdb;ipdb.set_trace()
        img = self.decoder(img)
        return img

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), learning_rate)
        # schedular = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        schedular = lr_scheduler.StepLR(optimizer, 10, gamma=.5)
        return [optimizer], [schedular]

    def training_step(self, batch, batch_idx):
        target, edges = batch
        pred = self(target)
        loss1 = torch.mean(self.Loss(pred, target) * edges)
        loss2 = 1 - torch.sum(metrics.ssim(pred, target)) / len(target)
        self.log("train mse loss", loss1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train ssim loss", loss2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss1 + loss2 * ssim_scale

    def validation_step(self, batch, batch_idx):
        target = batch
        pred = self(target)
        loss1 = torch.mean(self.Loss(pred, target))
        loss2 = 1 - torch.sum(metrics.ssim(pred, target)) / len(target)
        self.log('test mse loss', loss1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test ssim loss', loss2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
