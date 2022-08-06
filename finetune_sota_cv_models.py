from __future__ import print_function 
from __future__ import division
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
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
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

"""
Ref: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""


##### parameter setting #####
# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
parser = ArgumentParser()
# Requirement
parser.add_argument('--num_classes', type=int, default=75,
                    help="the num of classes for classification")
parser.add_argument('--data_dir', type=str, default='./dataset/', help='')
parser.add_argument('--model_name', type=str, default='vit_b_16', choices=['resnet', 'resnet101', 
'efficientnet_b5', 'efficientnet_b7', 'efficientnet_v2_m', 
'convnext_base', 'vit_b_16', 'swin_b', 
'alexnet', 'vgg', 'inception'], help='')
parser.add_argument('--batch_size', type=int, default=32, help="")
parser.add_argument('--num_epochs', type=int, default=50, help="")
parser.add_argument('--accum_iter', type=int, default=128, help="")
parser.add_argument('--feature_extract', type=bool, default=False, help="")

parser = parser.parse_args()

data_dir = parser.data_dir

'''
Models to choose from [resnet, resnet101, efficientnet_b5, efficientnet_b7, efficientnet_v2_m, 
convnext_base, vit_b_16, swin_b, 
alexnet, vgg, inception]
'''
model_name = parser.model_name

# Number of classes in the dataset
num_classes = parser.num_classes

# Batch size for training (change depending on how much memory you have)
batch_size = parser.batch_size

# Number of epochs to train for 
num_epochs = parser.num_epochs

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = parser.feature_extract

accum_iter = parser.accum_iter


##### define training #####
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss = loss / accum_iter
                        if phase == 'train':
                            loss.backward()

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train' and (((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dataloaders[phase]))):
                        optimizer.step()
                        optimizer.zero_grad()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                scheduler.step() # step at end of epoch

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, f"{model_name}_feature_extract_{feature_extract}.pt")
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


##### define testing #####
def test_model(model, dataloaders, is_inception=False):
    since = time.time()
    best_acc = 0.0   
    model.eval()   # Set model to evaluate mode
    running_corrects = 0
    phase = "test"
    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    best_acc = epoch_acc.item()
    print('{} Acc: {:.4f}'.format(phase, epoch_acc))
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    print()

    return model, best_acc


##### set parameter requires grad #####
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


##### Define Bloks #####
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


##### initialize model #####
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "efficientnet_b5":
        """ efficientnet_b5
        """
        model_ft = models.efficientnet_b5(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(2048, num_classes),
        )
        input_size = 224

    elif model_name == "efficientnet_b7":
        """ efficientnet_b7
        """
        model_ft = models.efficientnet_b7(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(2560, num_classes),
        )
        input_size = 224

    elif model_name == "efficientnet_v2_m":
        """ efficientnet_v2_m
        """
        model_ft = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1280, num_classes),
        )
        input_size = 224

    elif model_name == "vit_b_16":
        """ vit_b_16
        """
        model_ft = models.vit_b_16(weights='IMAGENET1K_V1')
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.heads = nn.Linear(768, num_classes, bias=True)
        input_size = 224

    elif model_name == "swin_b":
        """ swin_b
        """
        model_ft = models.swin_b(weights='IMAGENET1K_V1')
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.head = nn.Linear(1024, num_classes, bias=True)
        input_size = 238

    elif model_name == "convnext_base":
        """ convnext_base
        """
        model_ft = models.convnext_base(weights='IMAGENET1K_V1')
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier = nn.Sequential(
            LayerNorm2d((1024,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(1024, num_classes, bias=True),
        )
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)


##### Load Data #####
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

# Detect if we have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### Create the Optimizer #####
# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.AdamW(params_to_update, lr=0.001)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.97)


##### Run Training and Validation Step #####
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs, is_inception=(model_name=="inception"))
model_ft, test_result = test_model(model_ft, dataloaders_dict, is_inception=(model_name=="inception"))



##### Save Model #####
torch.save(model_ft, f"{model_name}_feature_extract_{feature_extract}.pt")

json_data = {"hist": hist, "best_val_acc": max(hist), "test_acc": test_result}
with open(f"{model_name}_feature_extract_{feature_extract}.json", "w") as f:
    json.dump(json_data, f, indent=6, ensure_ascii=False)