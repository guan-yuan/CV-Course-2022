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
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.font_manager as fm
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

# read a video frame by frame
vidcap = cv2.VideoCapture('dataset/pexels-anna-bondarenko-5757715.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("dataset/video_images/image"+str(count).zfill(3)+".jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 1/30 
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

# parse the image list
img_list = glob("dataset/video_images/*")
img_list.sort()
# print(img_list)

# load model and define transforms
model = torch.load("results/efficientnet_b3_feature_extract_False.pt").to(device)
# print(model)

input_size = 224

test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# define func to predict one image
def predict_image(image):
    image_tensor = test_transforms(image) # torch.Size([3, 224, 224])
    # print(image_tensor.shape)
    image_tensor = image_tensor.unsqueeze_(0) # torch.Size([1, 3, 224, 224])
    # print(image_tensor.shape)
    input = image_tensor
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

# load class name
with open("dataset/class_names.json") as f:
    class_name = json.load(f)

# predict
for img in img_list:
    copy_img = img
    img = Image.open(img).convert('RGB')
    # img.show()
    # print(type(img))
    print(class_name[predict_image(img)])
    img = Image.open(copy_img).convert('RGB')
    I1 = ImageDraw.Draw(img)
    fontsize = 100
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), fontsize)
    I1.text((28, 36), class_name[predict_image(img)], font=font, fill=(255, 255, 255))
    img.save(f"dataset/image2video/{copy_img.split(os.sep)[-1]}.jpg")



# convert to mp4 file
img_array = glob('dataset/image2video/*')
img_array.sort()
img_array_ = []
for filename in img_array:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array_.append(img)

_fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('dataset/demo.mp4', _fourcc, 20.0, size)
 
for i in range(len(img_array_)):
    out.write(img_array_[i])
out.release()