import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.optim import Adam, lr_scheduler
from torchjpeg import quantization, dct, metrics
import warnings


img_size = 256
warnings.filterwarnings('ignore')
batch_size = 20
learning_rate = 0.001
ssim_scale = 5


class Jpeg(nn.Module):
    def __init__(self, quality=10):
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


class DenseBlock(nn.Module):
    def __init__(self, num_layers, out_dim=3, k=3, p=0.):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(k * i, k, 3, padding='same') for i in range(1, num_layers)])
        self.relu = nn.LeakyReLU(inplace=True)
        self.cat_conv = nn.Conv2d(k*num_layers, out_dim, kernel_size=1)
        self.drop = nn.Dropout2d(p)

    def forward(self, x):
        for conv in self.layers:
            # import ipdb;ipdb.set_trace()
            out = self.drop(self.relu(conv(x)))
            x = torch.cat((x, out), dim=1)
        x = self.cat_conv(x)
        return x


class Subpix(nn.Module):
    def __init__(self, dim, scale=4):
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.up_conv = nn.Conv2d(dim, dim * scale**2, kernel_size=7, padding='same')
        self.conv = nn.Conv2d(dim * scale**2, dim * scale**2, kernel_size=3, padding='same')
        self.upScale = nn.PixelShuffle(scale)
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.up_conv(x))
        x = self.relu(self.conv(x))
        return self.upScale(x)


class DownScale(nn.Module):
    def __init__(self, dim, out_dim, scale=4):
        super().__init__()
        self.start_conv = nn.Conv2d(dim, dim*scale, 7, padding="same")
        self.down_conv = nn.Conv2d(dim*scale, dim*scale, 7, padding=3, stride=4)
        self.end_conv = nn.Conv2d(dim*scale, out_dim, 5, padding='same')
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.start_conv(x))
        x = self.relu(self.down_conv(x))
        x = torch.tanh(self.end_conv(x))
        return x


class UpScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.input_conv = nn.Conv2d(64, 8, kernel_size=5, padding='same')
        self.upScale = Subpix(dim=8)
        self.end_conv = nn.Conv2d(8, 3, kernel_size=5, padding='same')

    def forward(self, x):
        x = self.relu(self.input_conv(x))
        x = self.upScale(x)
        x = self.end_conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, p):
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout2d(p=p)
        self.start_conv = nn.Conv2d(3, 64, 7, padding='same')
        self.conv = nn.Conv2d(64, 64, 3, padding='same')
        self.input_conv = nn.Conv2d(64, 4, 5, padding='same')
        self.dense = DenseBlock(num_layers=num_layers, p=p, out_dim=64, k=4)
        self.cat_conv = nn.Conv2d(64, 16, 1)
        self.downScale = DownScale(dim=16, out_dim=3)
        # self.norm = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.relu(self.start_conv(x))
        res = self.conv(x)
        x = self.relu(self.input_conv(self.relu(res)))
        x = self.relu(self.dense(x) + res)
        x = self.cat_conv(x)
        x = self.downScale(x)
        return (x+1)/2


class Decoder(nn.Module):
    def __init__(self, num_layers, dense_layer, p, k=8):
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.start_conv = nn.Conv2d(3, 64, kernel_size=7, padding='same')
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.input_conv = nn.Conv2d(64, 8, kernel_size=5, padding='same')
        self.layers = nn.ModuleList([DenseBlock(dense_layer, k=k, out_dim=k, p=p) for _ in range(num_layers-1)])
        self.cat_conv = nn.Conv2d(num_layers*k, 64, kernel_size=1)
        self.upScale = UpScale()
        # self.norm = nn.BatchNorm2d(4)
        self.drop = nn.Dropout2d(p=p)

    def forward(self, x):
        x = self.drop(self.relu(self.start_conv(x)))
        res = self.conv(x)
        x = self.input_conv(res)
        out = x
        for layer in self.layers:
            out = layer(out)
            x = torch.cat((x, out), dim=1)
        x = self.drop(self.relu(self.cat_conv(x) + res))
        return torch.tanh(self.upScale(x))


class ImageCompression(LightningModule):
    def __init__(self, dense=16, p=0.1, q=50):
        super().__init__()
        self.Loss = nn.L1Loss(reduction="none")
        self.pixels = img_size**2 * batch_size * 3
        self.jpeg = Jpeg(quality=q)
        self.encoder = Encoder(dense, p=p)
        self.decoder = Decoder(dense//2, dense, p=p)
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
        schedular = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        # schedular = lr_scheduler.StepLR(optimizer, 10, gamma=.5)
        return [optimizer], [schedular]

    def training_step(self, batch, batch_idx):
        target, edges = batch
        pred = self(target)
        loss1 = torch.sum(self.Loss(pred, target) * edges) / self.pixels
        loss2 = 1 - torch.sum(metrics.ssim(pred, target)) / len(target)
        self.log("train mse loss", loss1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train ssim loss", loss2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss1 + loss2 * ssim_scale

    def validation_step(self, batch, batch_idx):
        target = batch
        pred = self(target)
        loss1 = torch.sum(self.Loss(pred, target)) / self.pixels
        loss2 = 1 - torch.sum(metrics.ssim(pred, target)) / len(target)
        self.log('test mse loss', loss1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test ssim loss', loss2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
