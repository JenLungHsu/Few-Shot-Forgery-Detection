import glob
import numpy as np
import torch, os, pdb
import random as rn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torch.multiprocessing
from PIL import Image
import cv2
import os, glob, numpy as np
from os.path import join as osj
from skimage.restoration import denoise_wavelet
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.utils import save_image
from skimage.feature import local_binary_pattern, hog
from utils import yaml_config_hook
from dataset_final import *

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torchvision
import argparse
import pandas as pd

# for face detect
from imutils import face_utils
import imutils
import dlib
import random

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn import metrics
from torch import optim as optim
import torch.nn as nn
import torch
from dataset import get_loader
from network import return_pytorch04_xception
from network.xception import xception
from dataset.dataset_fast import FaceForensics
from dataset.dataset_celebDF import CelebDF

# from args import get_args

import tqdm
import time

import sys
sys.path.append('..')
from DeepfakeBench.training.networks.efficientnetb4 import efficientnet_b4


start_time = time.time()

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description="SimCLR")
config = yaml_config_hook("./config.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))
parser.add_argument("--local_rank", type=int)
parser.add_argument("--deepfake", type=str, default=None)
parser.add_argument("--exp", type=int, default=None)
parser.add_argument("--model", type=str, default='efficientnet_b4')

args = parser.parse_args([])
# args = get_args()

os.environ['CUDA_VISIBLE_DEVICES']='0' 

def get_label(y):
    # 檢查是否所有元素都是1
    if torch.all(y == 1):
        return 1
    # 檢查是否所有元素都是0
    elif torch.all(y == 0):
        return 0
    else:
        raise ValueError("y的值既不全是1也不全是0")

def detection( args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda')
    # model = torch.load('/ssd6/Roy/LeGR_master_weihao/ckpt/efficientnetb4_flops0.47_1_pruned.t7')

    model = efficientnet_b4()

	# 删除 classifier 中的第二层 (即 model.classifier[1])
    in_features = model.classifier[1].in_features
    new_classifier = nn.Sequential(*[model.classifier[i] for i in range(len(model.classifier)) if i != 1])
    new_classifier = nn.Sequential(new_classifier, nn.Linear(in_features, 2))

	# 将新的 classifier 赋值回模型
    model.classifier = new_classifier
    # print(model.classifier)

    # 加载模型的状态字典
    state_dict = torch.load(args.pretrained_weight_path)

    # 去掉参数名称中的 "module" 字符串
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]  # 去掉 "module."
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # 使用新的状态字典更新模型
    model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()
    
    print("#"*80)
    print(f"Start training model on {args.train_file_path}")
    print(f"Testing on {args.test_file_path}")
    print("#"*80)

    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    # test_dataset = FaceForensics(args.root_dir, args.test_file_path, img_batch = args.test_img_batch, transform = transform)
    test_dataset = CelebDF(args.root_dir, args.test_file_path, img_batch = args.test_img_batch, transform = transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        # batch_size=args.batch_size if args.test_aug!=1 else 1,
        batch_size= args.test_img_batch, #剛好配對每部影片 16 frames
        shuffle=False,
        drop_last=False,
        num_workers=args.workers if args.test_aug!=1 else 1,
        pin_memory=False,
    )
    
    acc=0
    #pbar = tqdm.tqdm(zip(fs, lab), total=len(train_dataset))
    labs=[]
    preds=[]
    probability=[]
    for item in tqdm.tqdm(test_loader):
        #get video name
        x, y = item['image'].to(device), item['label'].to(device)

        output = model(x)
        post_function=nn.Softmax(dim=1)
        output = post_function(output)

        # Cast to desired
        _, prediction = torch.max(output, 1)    # argmax        # batch_size 個 0 or 1
        x = np.mean(np.array(prediction.detach().cpu().numpy()), axis=0)  

        prob = np.mean(np.array(output[:,1].detach().cpu().numpy()), axis=0) #一部影片是假的機率
        pred = 1 if x>0.5 else 0    #一部影片的預測結果
        lab = get_label(y)          #一部影片的真實結果

        probability.append(prob)
        preds.append(pred)
        labs.append(lab)

    # print('len(probability):',len(probability))
    # print('len(preds):',len(preds))
    # print('len(labs):',len(labs))
        
    preds_arr=np.array(preds)
    labs_arr= np.array(labs)
    
    accs = accuracy_score(labs_arr, preds_arr)
    f1 = f1_score(labs_arr, preds_arr, average='macro')
    re = recall_score(labs_arr, preds_arr, average='macro')
    pr = precision_score(labs_arr, preds_arr, average='macro')
    fpr, tpr, thresholds = metrics.roc_curve(labs_arr, probability, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)    

    print('accs:',accs,'f1:', f1,'re:', re,'pr:', pr,'roc_auc:', roc_auc)
    return accs, f1, re, pr, roc_auc

import pandas as pd
results = []
args.percent = 0
accs, f1, re, pr, roc_auc = detection(args)
results.append([accs, f1, re, pr, roc_auc])


df = pd.DataFrame(results)
# df.columns=['masked ratio', 'AUC', 'Accuracy', 'Recall', 'Precision', 'F1-Scoee']
df.columns=['Accuracy', 'F1-Score', 'Recall', 'Precision', 'AUC']
# df.to_csv('./result_csv/c23.csv', index=False)

end_time = time.time()
run_time = end_time - start_time
print('run_time:',run_time)