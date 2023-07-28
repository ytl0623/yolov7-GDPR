# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:40:50 2023

@author: Allen Yeh
"""

import matplotlib.pyplot as plt
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

import warnings
warnings.filterwarnings("ignore")

import datetime
import os
from mat4py import loadmat

import time

import gradio as gr

from tqdm import tqdm
from glob import glob

'''
1. encrypt_image() function

Input:
# image : M*N*3 RGB image
# Mask_locs : N*2 numpy array

Output:
# image : M*N*3 RGB image 加密完成的影像

------------------------------------------

2. decrypt_image() function

Input:
# image : M*N*3 RGB image
# datetime : 想要解密的時間

Output:
# image : M*N*3 RGB image 解密完成的影像
'''

def loadTable():
    table_path = './Key_table'
    Table_R,Table_G,Table_B = [],[],[]
    
    # load Table and Encrypt
    Path = os.path.join(table_path,'Table_R.mat') # Table Name
    Table_R = loadmat(Path) # load Table
    Table_R = Table_R['A'] # Table list
    Table_R = Table_R[1] # Take Encrypted pixel value
    
    return np.array(Table_R)
    
def find_indices(array):
    indices = np.argwhere(array == 1)
    return indices

def encrypt_image( image, mask_locs, map_table):
    '''
    加密影像
    parameter : image : 要加密的影像 ; mask_locs : 需要加密的pixel位置 ; T_R,T_G,T_B : 要使用的加密對照表
    Output : img : 加密後的影像
    '''
    
    # dateTime
    now = datetime.datetime.now(tz = datetime.timezone(datetime.timedelta(hours=8)))
    today = datetime.date.today()
    nowtime = now.strftime('%H-%M-%S').split('-')
    hour, minute, sec = nowtime[0], nowtime[1], nowtime[2]
    
    # read image
    img = image
    
    # Saving path
    save_img_dir = os.path.join('./output/',str(today))
    save_mask_dir = os.path.join('./mask_locs/',str(today))
    
    
    # encrypting
    pixel_to_encrypt = img[mask_locs[:, 0], mask_locs[:, 1]]
    img[mask_locs[:, 0], mask_locs[:, 1]] = map_table[pixel_to_encrypt]
    
    
    # 檢查有無當下日期資料夾 (img)
    if not os.path.exists(save_img_dir): 
        os.mkdir(save_img_dir)
        
    # 檢查有無當下時間(hr)資料夾 (img)
    save_img_dir = os.path.join(save_img_dir,hour)
    if not os.path.exists(save_img_dir): 
        os.mkdir(save_img_dir)
        
    # 檢查有無當下時間(min)資料夾 (img)
    save_img_dir = os.path.join(save_img_dir,minute)
    if not os.path.exists(save_img_dir): 
        os.mkdir(save_img_dir)
    
    # 檢查有無當下日期資料夾 (mask locs)
    if not os.path.exists(save_mask_dir): 
        os.mkdir(save_mask_dir)
    
    # 檢查有無當下時間(hr)資料夾 (mask locs)
    save_mask_dir = os.path.join(save_mask_dir,hour)
    if not os.path.exists(save_mask_dir): 
        os.mkdir(save_mask_dir)
        
    # 檢查有無當下時間(min)資料夾 (mask locs)
    save_mask_dir = os.path.join(save_mask_dir,minute)
    if not os.path.exists(save_mask_dir): 
        os.mkdir(save_mask_dir)
    
    
    
    # Save Encrypted image and mask location info.
    save_img_path = os.path.join(save_img_dir , sec) # 路徑 './output/今天日期/hour/minute/second.jpg'
    save_mask_path = os.path.join(save_mask_dir , sec) # 路徑 './output/今天日期/hour/minute/second.npy'
    
    # 存加密圖片
    cv2.imwrite( save_img_path + '.jpg' , img)  # 用png會太大
    
    # 存當下 mask location 資訊
    np.save(save_mask_path,mask_locs)  # 路徑 './output/今天日期/hour/minute/second.npy'
    
    return img # return Encrypted image


def decrypt_image(datetime , map_table): # 取 datetime : yyyy-mm-dd (GUI input?)
    # # Read Encrypted Image
    # img = encrypted_image
    
    '''
    param : datetime( ./output/yyyy-mm-dd/hr/min/ ) path
    return : decrypted image ( numpy array M*N*3 )
    '''
    
    # Convert map_table to list type
    map_table = list(map_table)
    
    # Get all encrypted image in designated folder
    dir_list = glob(datetime+"*")
    
    img = cv2.imread(dir_list[0])
    sz = img.shape
    decrypt_image = np.empty((sz[0],sz[1],sz[2]), dtype = int)
    
    for img_slice in dir_list:
        # image path and mask_locs path
        image_path = img_slice
        locs_path = image_path.replace('output','mask_locs')
        locs_path = locs_path.replace('jpg','npy')
        
        # save decrypt image path
        save_path = datetime.replace('output','decrypted_output')
        out_image = image_path.replace('output','decrypted_output')
        
        if os.path.exists(save_path):
            os.makedirs(save_path)
        
        # read image
        encrypted_img, mask_locs = cv2.imread(image_path), np.load(locs_path)
        decrypt_image = np.copy(encrypted_img)
        
        # Decrypting image
        for mask_ind in range(len(mask_locs)):
            
            # 第 mask_ind 個的mask location
            x , y = mask_locs[mask_ind][0] , mask_locs[mask_ind][1]
            
            # 根據 Table 解密pixel
            decrypt_image[x, y] = [map_table.index(decrypt_image[x][y][0]), map_table.index(decrypt_image[x][y][1]), map_table.index(decrypt_image[x][y][2])]
            
        # End of for loop
        
        cv2.imwrite(out_image, decrypt_image)
        return decrypt_image

    return decrypt_image # return decrypted image


