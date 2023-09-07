import torch
import os
from model import LDRNet
import lightning as pl
import torchvision.transforms as transforms
import cv2 as cv
import configs
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from augmentation import normal_transform

model = LDRNet(100)
model.load_from_checkpoint(checkpoint_path = "all/epoch=164-step=49995.ckpt")
del model.backbone_model.border
torch.save(model.backbone_model.state_dict(), "temp.pth")