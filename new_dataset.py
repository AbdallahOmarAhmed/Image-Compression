import os

import numpy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms
img_size = 256

class NewDataset(Dataset):
    def __init__(self, train):
        self.train = train
        self.path = '/home/abdallah/datasets/2/Cleaned_Images/Cleaned_Images/final/'
        self.images = os.listdir(self.path)
        self.basic_trans = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.images) - 5000 if self.train else 5000

    def __getitem__(self, idx):
        aug = self.transform
        if not self.train:
            aug = self.basic_trans
            idx *= -1
        image = Image.open(self.path + self.images[idx])
        img = aug(image)
        if self.train:
            img_gray = numpy.array(image.resize((img_size, img_size)).convert('L'))
            canny = torch.from_numpy(cv2.Canny(image=img_gray, threshold1=50, threshold2=50))
            sobel = torch.from_numpy(cv2.Sobel(src=img_gray, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=5))
            edges = (torch.sigmoid(sobel)+(canny/255)).unsqueeze(0) + 1
            return img, edges
        return img

if __name__ == '__main__':
    pil = transforms.ToPILImage()
    data = NewDataset(True)
    for i in range(1000):
        x, edge = data[i+8]
        # import ipdb;ipdb.set_trace()
        # cv2.imshow("img", edge)
        pil((x+1)/2).show()
        cv2.waitKey(0)
