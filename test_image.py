import argparse
import sys
from io import BytesIO

import cv2
# from lightning_logs.basic_model import ImageCompression
import numpy

from model import ImageCompression
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
from dataset import Celeb
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pickle
import imgaug.augmenters as iaa


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path')
args = parser.parse_args()

# image = cv2.imread(args.path, cv2.IMREAD_GRAYSCALE)


# model = ImageCompression()
# model.load_state_dict(torch.load('modelLast.pth'))
# model.eval()
# loss = torch.nn.MSELoss()

# transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
# pil = transforms.ToPILImage()
model = ImageCompression.load_from_checkpoint(
    'lightning_logs/version_0/epoch=32-step=58937.ckpt')
model.eval()
loss = torch.nn.MSELoss()
pil = transforms.ToPILImage()
totensor = transforms.ToTensor()


def save(img, path):
    f = open(path, "wb")
    pickle.dump(img, f)
    f.close()


if args.path == None:
    # import ipdb;ipdb.set_trace()
    train_data = Celeb(train=True)
    test_data = Celeb(train=False)
    train_dataLoader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_dataLoader = DataLoader(test_data, batch_size=1, shuffle=True)
    for x in test_dataLoader:
        images = x
        x0 = model(images)
        out = (x0+1)/2
        output = pil(out.squeeze(0))
        print("max :", out.max().item(), "min :", out.min().item())
        print('loss :', loss(x0, images))
        print('ssim :', ssim(x0, images, data_range=1, size_average=True))
        print("________________________\n")
        orig = pil((x.squeeze(0)+1)/2)
        show = np.hstack((np.array(orig), np.array(output)))
        opencvImage = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
        cv2.imshow('test', opencvImage)
        cv2.waitKey(0)

else:
    outputIoStream = BytesIO()
    orig = Image.open(args.path)
    x = totensor(orig).unsqueeze(0)
    # import ipdb;ipdb.set_trace()
    x = (x - 0.5)/0.5
    x = model.encoder(x)
    pil(x[0]).save("com.jpg", "JPEG", quality=50)
    # pil(x[0]).save("com.jpg", "JPEG", qtables=qtd)
    # img = Image.open('com1.jpg')
    # img0 = totensor(img).unsqueeze(0) - 0.5
    # out = (model.decoder(img0) + 1)/2


    # qt = model.jpeg.mat
    # qtd = {}
    # qtd[0] = qt[0, 0].to(torch.int).flatten().tolist()
    # qtd[1] = qt[0, 1].to(torch.int).flatten().tolist()
    # orig.save(outputIoStream, "JPEG", quality=10)
    # x2 = totensor(Image.open(outputIoStream))
    # orig.save("orig.jpg", "JPEG", quality=10)

    # jpeg = DiffJPEG(quality=10, differentiable=True)
    # jpeg.eval()
    # jpeg2 = DiffJPEG(quality=10, differentiable=False)
    # x1 = jpeg(x, width=width, height=height).squeeze(0)


    # print(loss(x1, x2))




    # orig = Image.open(args.path)
    # image = totensor(orig).unsqueeze(0)
    # z = pil(model.encoder(image).squeeze(0))
    # outputIoStream = BytesIO()
    # z.save(outputIoStream, "JPEG", quality=10)
    # # outputIoStream.seek(0)
    # img = torch.from_numpy(Image.open(outputIoStream)).unsqueeze(0)
    # out = model.decompress(img)
    # print(loss(out, image))
    # print('ssim :', ssim(out, image, data_range=1, size_average=True))
    # output = pil(out.squeeze(0))
    # output.show()

    # out = model(image).clamp(0, 1)
    # pik = model.compression(image)
    # save(img=pik, path="test.pkl")
    # print(loss(out, image))
    # print('ssim :', ssim(out, image, data_range=1, size_average=True))
    # output = pil(out.squeeze(0))
    # output.show()
