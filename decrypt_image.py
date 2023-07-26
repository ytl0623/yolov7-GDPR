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

def decrypt_image( encrypted_image , datetime ): # 取 datetime 部分還沒完成
    # Read Encrypted Image
    img = encrypted_image
    
    # load mask location
    mask_locs = np.load(datetime + '13.npy')
    
    # Path
    root_dir = './'
    table_path = './Key_table'
    Table_Name = ['Table_R.mat', 'Table_G.mat', 'Table_B.mat']
    Alphabet = ['A','B','C']
    Table_R,Table_G,Table_B = [],[],[]
    
    # load Table and Encrypt
    Path = os.path.join(table_path,Table_Name[0]) # Table Name
    Table_R = loadmat(Path) # load Table
    Table_R = Table_R[Alphabet[0]] # Table list
    Table_R = Table_R[1] # Take Encrypted pixel value
    
    Path = os.path.join(table_path,Table_Name[1]) # Table Name
    Table_G = loadmat(Path) # load Table
    Table_G = Table_G[Alphabet[1]] # Table list
    Table_G = Table_G[1] # Take Encrypted pixel value
    
    Path = os.path.join(table_path,Table_Name[2]) # Table Name
    Table_B = loadmat(Path) # load Table
    Table_B = Table_B[Alphabet[2]] # Table list
    Table_B = Table_B[1] # Take Encrypted pixel value
    
    # Decrypted image
    for mask_ind in range(len(mask_locs)):
        # 第 mask_ind 個的mask location
        x,y = mask_locs[mask_ind][0],mask_locs[mask_ind][1]
        
        # 根據 Table 解密pixel
        img[x][y][0], img[x][y][1], img[x][y][2] = Table_R.index(img[x][y][0]), Table_G.index(img[x][y][1]), Table_B.index(img[x][y][2])
    
    # End of for loop

    return img

def time( year, month, day, hour, minute ) :
    encrypt_img = cv2.imread( "./output/{}-0{}-{}/{}/{}/".format( year, month, day, hour, minute ) + "13.jpg" )
    decrypt_img = decrypt_image( encrypt_img, "./mask_locs/{}-0{}-{}/{}/{}/".format( year, month, day, hour, minute ) )
    return decrypt_img
    
if __name__ == "__main__":
    year = gr.Slider(2000, 2024, 2023, step=1, label='year', info='iou threshold for filtering the annotations')
    month = gr.Slider(1, 12, 7, step=1, label='month', info='iou threshold for filtering the annotations')
    day = gr.Slider(1, 31, 26, step=1, label='day', info='iou threshold for filtering the annotations')
    hour = gr.Slider(1, 24, 13, step=1, label='hour', info='iou threshold for filtering the annotations')
    minute = gr.Slider(1, 60, 16, step=1, label='minute', info='iou threshold for filtering the annotations')

    # It's not suitable for set example.
    # problem: 1. need "07" not "7" 2. need filename not [SECOND].jpg, [SECOND].npy
    # decrypt video needs decrypt whole directory
    demo = gr.Interface( fn = time, 
                         inputs = [year, month, day, hour, minute],                
                         outputs = "image" ).launch()   






























