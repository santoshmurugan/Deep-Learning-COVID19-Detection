import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path

def get_dataloaders(device):
    '''
    Return: (Training_TensorDataset, Test_TensorDataset)
    '''
    #Load all of the images into lists
    base_path = Path(__file__).parent
    covid_imgs_dir = "%s%s" % (base_path, '/data/CT_COVID')
    covid_imgs = [os.path.join(covid_imgs_dir, img_path) for img_path in os.listdir(covid_imgs_dir)]
    covid_labels = np.repeat(1, len(covid_imgs))

    non_covid_dir = "%s%s" % (base_path, '/data/CT_NonCOVID')
    non_covid_imgs = [os.path.join(non_covid_dir, img_path) for img_path in os.listdir(non_covid_dir)]
    non_covid_labels = np.repeat(0, len(non_covid_imgs))

    all_imgs = covid_imgs + non_covid_imgs 
    labels = np.append(covid_labels,non_covid_labels)

    #Do a 70/30 train tests split
    X_train, X_test, y_train, y_test = train_test_split(all_imgs, labels, test_size=0.3, random_state=42)

    train_data = [np.array(transform(get_image(img))) for img in X_train]
    test_data = [np.array(transform(get_image(img))) for img in X_test]

    train_data = torch.tensor(train_data, dtype=torch.float32, device=device)
    test_data = torch.tensor(test_data, dtype=torch.float32, device=device)
    
    train_labels = torch.tensor(y_train, dtype=torch.float32, device=device)
    test_labels = torch.tensor(y_test, dtype=torch.float32, device=device)
    
    train_data = TensorDataset(train_data, train_labels)
    test_data = TensorDataset(test_data, test_labels)

    return train_data, test_data

def transform(image):

    #Note: This is the normalization needed for AlexNet, may need to change depending on model 
    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    return train_transformer(image)

def get_image(file_path):
    image = Image.open(file_path).convert("RGB") #Turns image to RGB as required
    return image


