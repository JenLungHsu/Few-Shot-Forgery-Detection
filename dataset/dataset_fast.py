import cv2
import numpy as np
import os
import random
import glob

import dlib
import torch
from PIL import Image
from torch.utils.data import Dataset
random.seed(42)

class FaceForensics(Dataset):
    def __init__(self, root_dir, file_path, img_batch = None, transform = None):
        super().__init__()  
        self.root_dir = root_dir
        self.img_batch = img_batch
        self.transform = transform

        self.image_size = 256

        with open(file_path, "r") as f:
            lines = f.read()
            self.video_list = [[i.split(' ')[0], i.split(' ')[1]] for i in lines.split('\n')]
            random.shuffle(self.video_list)

            self.image_list = []

            # train, val
            if self.img_batch is None:
                for video in self.video_list:
                    img_path = os.path.join(self.root_dir, video[0], '*.png')
                    # print('img_path:',img_path)
                    img_list = glob.glob(img_path)
                    for i in img_list:
                        self.image_list.append([i, video[1]])
            # test
            else:
                for video in self.video_list:
                    img_path = os.path.join(self.root_dir, video[0], '*.png')
                    # print('img_path:',img_path)
                    img_list = glob.glob(img_path)
                    for i in random.sample(img_list, self.img_batch): # 取出不放回
                        self.image_list.append([i, video[1]])
                # print('self.image_list:',self.image_list)

            print('init:',len(self.image_list))

        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):

        img_path, label = self.image_list[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(img) == 0:
            return {'image': None, 'label': None ,'filename': img_path}
        
        # img = Image.fromarray(img)
        img = self.transform(img)
        label = int(label)
        label = torch.tensor(label)
            
        return {'image': img, 'label': label, 'filename': img_path}