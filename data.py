import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning as pl
import cv2 as cv
from tqdm import tqdm
import json
from augmentation import augment, normal_transform

def json_load(path):
    with open(path, "r") as f:
        return json.load(f)

def rearrange_points(l):
    l = sorted(l, key = lambda c: c[1])
    l[:2] = sorted(l[:2], key = lambda c: c[0])
    l[2:4] = sorted(l[2:4], key = lambda c: c[0], reverse = True)
    
    output = [x/224.0 for sublist in l for x in sublist]
    
    return output    

class DocDataModule(pl.LightningDataModule):
    def __init__(self, train_json_path, valid_json_path, data_dir, batch_size, num_workers, load_into_ram = False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_json_data = json_load(train_json_path)
        self.valid_json_data = json_load(valid_json_path)
        self.data_dir = data_dir
        self.load_into_ram = load_into_ram
    
    def setup(self, stage):
#         n_samples = len(self.json_data)
        
#         n_train_samples = round(n_samples*0.8)
#         n_valid_test_samples = n_samples - n_train_samples
#         n_valid_samples = round(n_valid_test_samples*0.5)
#         n_test_samples = n_valid_test_samples - n_valid_samples
        
#         entire_dataset = DocDataset(self.json_data, self.data_dir, self.load_into_ram)
        
#         self.train_dataset, valid_test_dataset = random_split(entire_dataset, [n_train_samples, n_valid_test_samples])
#         self.valid_dataset, self.test_dataset = random_split(valid_test_dataset, [n_valid_samples, n_test_samples])

        self.train_dataset = DocDataset(self.train_json_data, self.data_dir, transform = augment, load_into_ram = self.load_into_ram)
        self.valid_dataset = DocDataset(self.valid_json_data, self.data_dir, load_into_ram = self.load_into_ram)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False
        )

class DocDataset(Dataset):
    def __init__(self, data, data_dir, transform = normal_transform, load_into_ram = False):
        super().__init__()

        self.data = data
        print(len(self.data))
        self.data_dir = data_dir
        self.transform = transform
        self.load_into_ram = load_into_ram
        if self.load_into_ram:
            self.new_data = self.load_data_into_ram(self.data, "Loading data")
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # image = self.load_image(self.data_list[index]['image_path'])
        # mask = self.load_image(self.data_list[index]['mask_path'])
        if self.load_into_ram:
            return self.new_data[index]
        return self.load_image(self.data[index]['image_path'], self.data[index]['corners'])
    
    # don't use this
    def load_data_into_ram(self, data_list, ms):
        temp = []
        print(ms)
        for sample in tqdm(data_list):
            temp.append((self.load_image(sample['image_path']), torch.tensor(sample['corners'])))
        return temp
    
    def load_image(self, path, corners):
        path = self.data_dir +"/"+ path
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # image = transforms.ToTensor()(image)
        # image = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)
        
        transformed = self.transform(image=image, keypoints = corners)
        transformed_image = transformed["image"]
        transformed_corners = rearrange_points(transformed["keypoints"])
        
        return transformed_image, torch.tensor(transformed_corners, dtype = torch.float)