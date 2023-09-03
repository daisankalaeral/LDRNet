import torch
import os
from model import LDRNet
import lightning as pl
import torchvision.transforms as transforms
import cv2 as cv
import configs
from tqdm import tqdm

def load_image(image_path):
    path = image_path
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (244,244))
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)
    return image, image.shape

model = LDRNet(configs.n_points, lr = configs.lr)
model = model.load_from_checkpoint("good_ckpt/epoch=79-step=18240.ckpt")
model.eval()

input_dir = "15_images_for_test"
output_dir = "output_images"
for f in tqdm(os.listdir(input_dir)):
    image_path = input_dir + "/" + f
    image, _ = load_image(image_path)
    image = image.cuda()
    corners, _ = model(image.unsqueeze(0))
    
    output_image_path = output_dir + "/" + f
    
    img = cv.imread(image_path)
    img = cv.resize(img, (244,244))
    corners = corners[0].detach().cpu().numpy()
    x = corners[0::2] * img.shape[1]
    y = corners[1::2] * img.shape[0]
    
    colors = [(0,0,255), (0,255,0), (255,0,0), (255,0,255)]
    for i, (a,b) in enumerate(zip(x,y)):
        # s = (int(corners[i % 4][0]*img.shape[1]), int(corners[i % 4][1]*img.shape[0]))
        # e = (int(corners[(i+1) % 4][0]*img.shape[1]), int(corners[(i+1) % 4][1]*img.shape[0]))
        # img = cv.line(img, s, e, (0,0,255), 4)
        img = cv.circle(img, (int(a),int(b)), 3, colors[i], 2)
        
    cv.imwrite(output_image_path, img)