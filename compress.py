import argparse

from PIL import Image
from torchvision import transforms

from model import ImageCompression

parser = argparse.ArgumentParser()
parser.add_argument('imgPath')
parser.add_argument('outPath')
args = parser.parse_args()

model = ImageCompression.load_from_checkpoint('lightning_logs/version_0/epoch=32-step=58937.ckpt')
model.eval()

pil = transforms.ToPILImage()
totensor = transforms.ToTensor()

orig = Image.open(args.imgPath)
x = totensor(orig).unsqueeze(0)
x = (x - 0.5)/0.5
x = model.encoder(x)
pil(x[0]).save(args.outPath, "JPEG", quality=50)
