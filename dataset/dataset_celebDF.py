import cv2
import numpy as np
import os
import random
import glob

import dlib
import torch
from PIL import Image
from torch.utils.data import Dataset


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def get_face_crop(face_detector, image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray, 1)
    
    height, width = image.shape[:2]

    if len(faces) == 0:
        return None
    else:
        face = faces[0]
        x, y, size = get_boundingbox(face, width, height)

        cropped_face = image[y:y + size, x:x + size]
        return cropped_face


def load_and_preprocess_image(image_filename, output_image_size, face_detector, save_path=None):
    image = cv2.imread(image_filename)
    # -----
    if image is None:
        return None
    else:
    # -----
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        cropped_image = get_face_crop(face_detector, image)
        if cropped_image is None:
            return None
    
        resized_image = cv2.resize(cropped_image, (output_image_size, output_image_size)) # 轉回原圖大小
        return resized_image

class CelebDF(Dataset):
    def __init__(self, root_dir, file_path, img_batch = None, transform = None):
        super().__init__()  
        self.root_dir = root_dir
        self.img_batch = img_batch
        self.transform = transform

        self.image_size = 256
        self.face_detector = dlib.get_frontal_face_detector()

        with open(file_path, "r") as f:
            lines = f.read()
            self.video_list = [[i.split(' ')[0], i.split(' ')[1]] for i in lines.split('\n')]
            random.shuffle(self.video_list)

            self.image_list = []

            # train, val
            if self.img_batch is None:
                for video in self.video_list:
                    img_path = os.path.join(self.root_dir, video[0], '*.png')
                    # print(img_path)
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
        # img = cv2.imread(img_path)
        img = load_and_preprocess_image(img_path, self.image_size, self.face_detector)
        
        if img is None:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = []
            # img = np.zeros((299, 299, 3), dtype=np.uint8)  # 填充为空图像
            # print('Image is None')

        if len(img) == 0:
            return {'image': None, 'label': None ,'filename': img_path}
        
        # img = Image.fromarray(img)
        img = self.transform(img)
        label = int(label)
        label = torch.tensor(label)
            
        return {'image': img, 'label': label, 'filename': img_path}
    



class CelebDF_fewimage(Dataset):
    def __init__(self, root_dir, file_path, img_batch = None, transform = None):
        super().__init__()  
        self.root_dir = root_dir
        self.img_batch = img_batch
        self.transform = transform

        self.image_size = 256
        self.face_detector = dlib.get_frontal_face_detector()

        with open(file_path, "r") as f:
            lines = f.read()
            self.video_list = [[i.split(' ')[0], i.split(' ')[1]] for i in lines.split('\n')]
            random.shuffle(self.video_list)

            self.image_list = []

            # train, val
            if self.img_batch is None:
                for video in self.video_list:
                    img_path = os.path.join(self.root_dir, video[0])
                    img_path += '.jpg'
                    # print('img_path:',img_path)

                    self.image_list.append([img_path, video[1]])
                    # print('self.image_list:',self.image_list)
            # test
            else:
                for video in self.video_list:
                    img_path = os.path.join(self.root_dir, video[0])
                    img_path += '.jpg'
                    # print('img_path:',img_path)

                    self.image_list.append([img_path, video[1]])
                    # print('self.image_list:',self.image_list)

            print('init:',len(self.image_list))

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):

        img_path, label = self.image_list[index]
        # print('img_path:',img_path)
        # img = cv2.imread(img_path)
        # print(img)
        img = load_and_preprocess_image(img_path, self.image_size, self.face_detector)
        
        if img is None:
        #     img = cv2.imread(img_path)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = []
            img = np.zeros((299, 299, 3), dtype=np.uint8)  # 填充为空图像
            # print('Image is None')

        if len(img) == 0:
            return {'image': None, 'label': None ,'filename': img_path}
        
        # img = Image.fromarray(img)
        img = self.transform(img)
        label = int(label)
        label = torch.tensor(label)
            
        return {'image': img, 'label': label, 'filename': img_path}
    

def get_subdirectories(folder):
    subdirectories = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    return subdirectories

class CelebDF_JYUNYI(Dataset):
    def __init__(self, root_dir, file_path, img_batch = None, transform = None):
        super().__init__()  
        self.root_dir = root_dir
        self.img_batch = img_batch
        self.transform = transform

        self.image_size = 256
        self.face_detector = dlib.get_frontal_face_detector()
        self.video_list = []
        self.image_list = []

        for name in ['Fake', 'Real']:
            folder_path = os.path.join(file_path, name)
            subdirectories = get_subdirectories(folder_path)
            for subdir in subdirectories:
                img_path = os.path.join(folder_path, subdir, '*.png')
                img_list = glob.glob(img_path)

                # train, val
                if self.img_batch is None:
                    for i in img_list:
                        self.image_list.append([i, '1' if name == 'Fake' else 0])
                # test
                else:
                    for i in random.sample(img_list, self.img_batch): # 取出不放回
                        self.image_list.append([i, '1' if name == 'Fake' else 0])
        # print(self.image_list)
        print('init:',len(self.image_list))

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):

        img_path, label = self.image_list[index]
        img = cv2.imread(img_path)
        # img = load_and_preprocess_image(img_path, self.image_size, self.face_detector)
        
        if img is None:
            # img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = []
            img = np.zeros((299, 299, 3), dtype=np.uint8)  # 填充为空图像
            print('Image is None')

        if len(img) == 0:
            return {'image': None, 'label': None ,'filename': img_path}
        
        # img = Image.fromarray(img)
        img = self.transform(img)
        label = int(label)
        label = torch.tensor(label)
            
        return {'image': img, 'label': label, 'filename': img_path}