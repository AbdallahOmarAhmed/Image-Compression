import sys
import numpy
from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms

sys.path.append("/home/abdallah/projects/Image-Compression/JPEG")
from DiffJPEG import DiffJPEG

loss = torch.nn.MSELoss()
pil = transforms.ToPILImage()
totensor = transforms.ToTensor()


outputIoStream = BytesIO()
orig = Image.open("test.jpg")

orig.save(outputIoStream, "JPEG", quality=10)
x2 = totensor(Image.open(outputIoStream))

width, height = orig.size
x = totensor(orig).unsqueeze(0)
jpeg = DiffJPEG(quality=10, differentiable=True)
jpeg2 = DiffJPEG(quality=10, differentiable=False)
jpeg.eval()
# import ipdb;ipdb.set_trace()
x1 = jpeg(x, width=width, height=height).squeeze(0)



# y, cb, cr = jpeg.compressor(x, width=width, height=height)
# y = pil(y)
# cb = pil(cb)
# cr = pil(cr)
# img = Image.merge('YCbCr', (y, cb, cr))
print(loss(x1, x2))

# image = Image.open("tested.jpg")
# outputIoStream = BytesIO()
# image.save(outputIoStream, "JPEG", quality=100)
# x2 = Image.open(outputIoStream)
# ycbcr = x2.convert('YCbCr')
# B = numpy.ndarray((image.size[1], image.size[0], 3), 'u1', ycbcr.tobytes())
# Z = numpy.array(image)
# import ipdb;ipdb.set_trace()

# m = np.array(y)
