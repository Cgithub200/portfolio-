import csv
import io
import torch.nn as nn
import json
from os import walk
from pathlib import Path
import random
import numpy as np 
from PIL import Image as im 
import os, glob
import torch
import cv2
import torch
import numpy as np
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
from datasets import Dataset as ds
from torchvision import transforms
import os
import pandas as pd
import torch.nn.functional as functional
from PIL import Image
import os
from fuzzywuzzy import fuzz
from transformers import BertTokenizer,DistilBertTokenizer

version = "b0"
Effsizes = {
            'b0': (224, 224), 'b1': (240, 240), 
            'b2': (260, 260), 'b3': (300, 300),
            'b4': (380, 380), 'b5': (456, 456),
            'b6': (528, 528), 'b7': (600, 600),
        }


search_terms = [
    "cardiomegaly", "infiltration", "nodule", "emphysema", "effusion", 
    "atelectasis", "pneumothorax", "mass", "fibrosis", "consolidation", 
    "edema", "pneumonia", "hypertension", "neoplasm", "abscess", "cyst", 
    "hernia", "hemorrhage", "ischemia", "lesion", "osteoporosis", 
    "scoliosis", "stenosis", "thrombosis", "infarction", "aneurysm", 
     "arthritis", "fracture", "degeneration","stricture"
]

"""
search_terms = [
            "cardiomegaly",
            "infiltration",    #
            "nodule",
            "emphysema",  #
            "effusion",    #
            "atelectasis",  
            "pneumothorax",  #
            #######"mass",
            "fibrosis",
            "consolidation",
            "edema",   
            "pneumonia",
]
"""

def get_image_names(directory):
    files = []
    for filename in walk(directory):
        lowercase_filename = [element.lower() for element in filename[2]]
        files.extend(lowercase_filename)
        break
    return files

def extractcsv(csv_file_name):
    #csv_path = os.path.join(os.path.dirname(__file__), 'csv_file_name')
    with open('csv_file_name') as csv_file:
        reader = csv.reader(csv_file)
        rows = []
        for row in reader:
            rows.append(row) 
        csv_file.close()
    return rows

def writecsv(rows,csv_file_name): 
    with open(csv_file_name, 'w', newline='', encoding="utf-8")as csv_file:
        writer = csv.writer(csv_file, delimiter=',',)
        for row in rows:
            writer.writerow(row)

def resize_images(input_folder, output_folder, target_size=(224, 224), file_format='PNG'):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all files in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            # Construct input and output paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.' + file_format.lower())
            
            # Check if the output file already exists
            if not os.path.exists(output_path):
                try:
                    # Open the image
                    with Image.open(input_path) as img:
                        # Check if the image is black and white, if so, convert to RGB
                        if img.mode == 'L':
                            img = img.convert('RGB')
                        # Resize the image
                        img_resized = img.resize(target_size)
                        # Save the resized image as PNG format
                        img_resized.save(output_path, format=file_format)
                        print(f"Resized and saved: {output_path}")
                except Exception as e:
                    print(f"Error processing image '{filename}': {e}")
            else:
                print(f"Skipping '{filename}' as it already exists in the output folder")

def is_similar_to_any(word, word_array, threshold=80):
    for candidate in word_array:
        if fuzz.token_sort_ratio(word, candidate) >= threshold:
            return True
    return False



def replace_with_space(error):
    return (' ', error.end)

def data_cleanup():
    current_path = os.getcwd()
    my_classes = []
    max_indices = 0
    max_negative = 0
    max_limit = 300
    roco_files = get_image_names(f"{current_path}/my_clip/static/roco-dataset/data/train/radiology/images")

    rows = [["id","Image Index","Label","label_map","captions"]]
    with open(f"{current_path}/my_clip/static/roco-dataset/data/train/radiology/captions.txt", 'r', encoding='utf-8') as file:
        
        lines = file.readlines()
        
        for counter,line in enumerate(lines):
            line = ''.join(c if ord(c) < 128 else '' for c in line)
            line = [element.strip('\n').lower() for element in line.split('\t') if element.strip()]
            line[0]+='.png'
            try:
                elements = line[1].split(' ')
                elements = [item.lower() for item in search_terms if is_similar_to_any(item.lower(), elements, threshold=95)]
                elements = list(set(elements))
                my_classes += [element for element in elements if element not in my_classes]
                indices = [my_classes.index(element) for element in elements if element in my_classes]
                if len(indices) > max_indices:
                    max_indices = len(indices)

                if (counter % 1000 == 999):
                    print(f"counter {counter + 1} my_classes {my_classes}")
                    
                if (counter > 10000):
                    break
            except:
                print(f"issue {line}")
        class_frequency = {element: 0 for element in my_classes}
        for counter,line in enumerate(lines):
            line = ''.join(c if ord(c) < 128 else '' for c in line)
            line = [element.strip('\n').lower() for element in line.split('\t') if element.strip()]
            line[0]+='.png'
            try:
                
                elements = line[1].split(' ')
                elements = [item.lower() for item in search_terms if is_similar_to_any(item.lower(), elements, threshold=95)]
                elements = list(set(elements))
                indices = []
                continue_outer_loop = False
                for element in elements:
                    if element in my_classes:
                        #print(f"here {my_classes}")
                        #print(f"element {element} class_frequency {class_frequency}")
                        class_frequency[element] += 1
                        if class_frequency[element] > max_limit:
                            continue_outer_loop = True 
                            break
                        indices += [my_classes.index(element)]
                if continue_outer_loop:
                    continue

                #indices = [my_classes.index(element) for element in elements if element in my_classes]
                indices_str = "|".join(str(index) for index in indices)
                binary_condtions = binary_classification_conv(indices_str.split("|"),len(my_classes)) if len(indices) > 0 else ''

                for i in range(max_indices - len(indices)):
                    indices_str += "|-1"
                if len(indices) == 0:
                    indices_str = indices_str[1:]
                    sequence = "0|"
                    binary_condtions = (sequence * len(my_classes))[:-1]

                if line[0] not in roco_files:
                    continue
                line[0] = line[0].upper()
                
                if len(indices) == 0:
                    if max_negative < 1:
                        continue
                    else:
                        max_negative -= 1

                line = [counter] + line[:1] + [indices_str] + [binary_condtions] + line[1:]
                rows.append(line)
                if (counter % 500 == 499):
                    print(f"counter {counter + 1}")

            except:
                print(f"issue {line}")
    print(class_frequency)
    print("done")
    writecsv(rows,'temp_2.csv')
    exit()
    

import xml.etree.ElementTree as ET
import requests






import albumentations
dir_path = os.path.join(os.getcwd(), "my_clip" , "static" , "roco-dataset", "data", "train", "radiology", "images")
class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, label_map , labels , train = True , max_seq_length = 250):
        self.image_filenames = image_filenames
        self.captions = list(captions)

        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=max_seq_length
        )
        self.labels = labels
        self.label_map = label_map
        self.resize = albumentations.Resize(224, 224, always_apply=True)
        self.Norm = albumentations.Normalize(max_pixel_value=255.0, always_apply=True)
        self.train = train
        self.augmentation = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0, 0.5)),  # blur images with a sigma of 0 to 0.5
            iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255)),  # add gaussian noise
            iaa.Multiply((0.9, 1.1), per_channel=0.1),  # multiply pixel values
            iaa.Affine(translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}),
        ], random_order=True)  # apply augmentations in random order

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        image = im.open(f"{dir_path}/{self.image_filenames[idx]}") if dir else None
        image = self.resize(image = np.array(image))["image"]
        if self.train:
            image = self.augmentation(image=image)                          #This destroys it
        image = self.Norm(image=image)["image"]

        image = torch.tensor(image).permute(2, 0, 1).float()
        item['image'] = image
        item['caption'] = self.captions[idx]
        item['label'] = torch.tensor([float(x) for x in (self.labels[idx]).split("|")])
        item['label_map'] = torch.tensor([float(x) for x in (self.label_map[idx]).split("|")])
        item['image_filenames'] = self.image_filenames[idx]
        return item
    def __len__(self):
        return len(self.captions)


def get_dfs(split_ratio = 0.2):
    dataframe = pd.read_csv(os.path.join(os.path.dirname(__file__), 'temp.csv'))
    max_id = dataframe["id"].max() + 1
    image_ids = np.arange(0, max_id)
    np.random.seed(10)
    valid_ids = np.random.choice(image_ids, size=int(split_ratio * len(image_ids)), replace=False)
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]

    test_ids = np.random.choice(train_ids, size=int(split_ratio * len(train_ids)), replace=False)
    train_ids = [id_ for id_ in train_ids if id_ not in test_ids]
    train_set = set(train_ids)
    valid_set = set(valid_ids)
    test_set = set(test_ids)
    if train_set.intersection(valid_set) or train_set.intersection(test_set) or valid_set.intersection(test_set):
        print("contamination")
    else:
        print("no contamination")

    print(f"")

    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    test_dataframe = dataframe[dataframe["id"].isin(test_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe , test_dataframe

def get_class_weights(dataframe):
    length = len(dataframe)
    weights = {}
    for i in range(length):
        classes = str(dataframe['Label'][i]).split("|")
        for my_class in classes:     
            my_class = int(my_class)
            if my_class == -1:
                break  
            if my_class in weights:
                weights[my_class] += 1
            else:
                weights[my_class] = 1   
    weights = {k: weights[k] for k in sorted(weights.keys())}
    weights = [float(weight) for weight in weights.values()]
    #weights = weights[:-1]   #Remove no class
    total_sum = sum(weights)
    weights = [1/(weight/total_sum) for weight in weights]
    weights = [weight/np.mean(weights) for weight in weights]
    print(f"weights {weights}")
    return weights


def create_loaders(dataframe, batch_size, train = True):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = Dataset(
        dataframe["Image Index"].values,
        dataframe["captions"].values,
        tokenizer=tokenizer,
        labels = dataframe["Label"].values,
        label_map = dataframe["label_map"].values,
        train=train,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= batch_size,
        shuffle=train,
    )
    return dataloader

def binary_classification_conv(targets,num_classes):
    targets2 = torch.zeros(num_classes)
    targets2[[int(a) for a in targets]] = 1
    tensor_string = [str(int(x.item())) for x in targets2]
    targets2 = '|'.join(tensor_string)
    return targets2
        
if __name__ == "__main__":
    user_response = input("Text =t or image = i or r = resize ")
    if user_response == "i":
        data_cleanup()
    elif user_response == "r":
        current_path = os.getcwd()
        input = f"{current_path}/my_clip/static/roco-dataset/data/train/radiology/images_"
        output = f"{current_path}/my_clip/static/roco-dataset/data/train/radiology/images"
        resize_images(input, output)


        