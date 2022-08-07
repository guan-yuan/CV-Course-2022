import cv2
from glob import glob
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import json
from argparse import ArgumentParser
from PIL import Image
# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def set_seed(seed=2022, loader=None):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass

set_seed()
# Detect if we have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # read a video frame by frame
# vidcap = cv2.VideoCapture('dataset/pexels-anna-bondarenko-5757715.mp4')
# def getFrame(sec):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#     hasFrames,image = vidcap.read()
#     if hasFrames:
#         cv2.imwrite("dataset/video_images/image"+str(count).zfill(3)+".jpg", image)     # save frame as JPG file
#     return hasFrames
# sec = 0
# frameRate = 0.5 #//it will capture image in each 0.5 second
# count=1
# success = getFrame(sec)
# while success:
#     count = count + 1
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success = getFrame(sec)

img_list = glob("dataset/video_images/*")
img_list.sort()
# print(img_list)

model = torch.load("results/efficientnet_b3_feature_extract_False.pt").to(device)
# print(model)

input_size = 224

test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def predict_image(image):
    image_tensor = test_transforms(image) # torch.Size([3, 224, 224])
    print(image_tensor.shape)
    image_tensor = image_tensor.unsqueeze_(0) # torch.Size([1, 3, 224, 224])
    print("="*10)
    print(image_tensor.shape)
    input = image_tensor
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

for img in img_list:
    img = Image.open(img).convert('RGB')
    img.show()
    print(type(img))
    print(predict_image(img))
    exit()