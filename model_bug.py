import math

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, lr_scheduler
from torchjpeg import quantization, dct, metrics, codec
import warnings

warnings.filterwarnings('ignore')
batch_size = 16
learning_rate = 0.001
# learning_rate = 1e-4
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


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
            out = self.drop(self.relu(conv(x)))
            x = torch.cat((x, out), dim=1)
        x = self.cat_conv(x)
        return x


class Subpix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.up_conv = nn.Conv2d(dim, dim * 16, kernel_size=7, padding='same')
        self.conv = nn.Conv2d(dim * 16, dim * 16, kernel_size=3, padding='same')
        self.upScale = nn.PixelShuffle(4)
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
        self.end_conv = nn.Conv2d(dim*scale, out_dim, 1)
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
        self.input_conv = nn.Conv2d(64, 8, kernel_size=7, padding='same')
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
        self.input_conv = nn.Conv2d(64, 4, 3, padding='same')
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
    def __init__(self, num_layers, dense_layer, p):
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.start_conv = nn.Conv2d(3, 64, kernel_size=7, padding='same')
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.input_conv = nn.Conv2d(64, 4, kernel_size=5, padding='same')
        self.layers = nn.ModuleList([DenseBlock(dense_layer, k=4, out_dim=4, p=p) for _ in range(num_layers-1)])
        self.cat_conv = nn.Conv2d(num_layers*4, 64, kernel_size=1)
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
    def __init__(self, dense=16, p=0.1, q=20):
        super().__init__()
        self.Loss = nn.L1Loss()
        self.jpeg = Jpeg(quality=q)
        self.encoder = Encoder(dense, p=p)
        self.decoder = Decoder(12, dense, p=p)
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
        pred = self(batch)
        loss1 = self.Loss(pred, batch)
        # import ipdb;ipdb.set_trace()
        loss2 = 1 - sum(metrics.ssim(pred, batch)) / len(pred)
        # loss = loss1 + loss2/ssim_scale
        # if math.isnan(loss) or math.isinf(loss):
        #     import ipdb;ipdb.set_trace()
        # self.log("train loss all", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train mse loss", loss1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train ssim loss", loss2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss1 + loss2

    def validation_step(self, batch, batch_idx):
        target = batch
        pred = self(batch)
        loss1 = self.Loss(pred, target)
        loss2 = 1 - sum(metrics.ssim(pred, target)) / len(batch)
        # loss = loss1 + loss2/ssim_scale
        # self.log('test loss all', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test mse loss', loss1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test ssim loss', loss2, on_step=False, on_epoch=True, prog_bar=False, logger=True)

# if __name__ == '__main__':
    # totensor = transforms.ToTensor()
    # model = ImageCompression()
    # orig = Image.open("test.jpg")
    # image = totensor(orig).unsqueeze(0)
    # x1, x2, x3 = model.compression(image)
    # import ipdb;ipdb.set_trace()

    # train_data = Celeb(train=True)
    # test_data = Celeb(train=False)
    # train_dataLoader = DataLoader(train_data, batch_size=1, shuffle=True)
    # test_dataLoader = DataLoader(test_data, batch_size=1, shuffle=False)
    # model = ImageCompression()
    # model = ImageCompression.load_from_checkpoint(
    #     'lightning_logs/full_data_set_18_layer_schduler_basic/checkpoints/epoch=29-step=62519.ckpt')
    # model.eval()
    # loss = torch.nn.MSELoss()
    # pil = transforms.ToPILImage()
    # for x in test_dataLoader:
    #     images = x
    #     out = model(images).clamp(0, 1)
    #     output = pil(out.squeeze(0))
    #     print(loss(out, images))
    #     orig = pil(x.squeeze(0))
    #     show = np.hstack((np.array(orig), np.array(output)))
    #     opencvImage = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
    #     cv2.imshow('test', opencvImage)
    #     cv2.waitKey(0)

# full_data_8_layer_schduler_basic_no_grad v0
# full_data_18_layer_schduler_basic_no_grad v3
# full_data_18_layer_schduler_basic_no_grad_drob_2 v4

