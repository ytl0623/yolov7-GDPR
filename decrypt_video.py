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

import argparse

from tqdm import tqdm
from glob import glob

from function import encrypt_image, decrypt_image, loadTable, find_indices, decrypt_video
 
def decrypt( year, month, day, hour, minute ) :
    print( "\nStart\n" )
    start_time = time.time()
    
    if ( int(month) < 10 ) :
        month = "0" + str( month )
    if ( int(day) < 10 ) :
        day = "0" + str( day )
    if ( int(hour) < 10 ) :
        hour = "0" + str( hour )
    if ( int(minute) < 10 ) :
        minute = "0" + str( minute )
    
    datetime = "./encrypted_output/{}-{}-{}/{}/{}/".format( year, month, day, hour, minute )
    decrypt_vid = decrypt_video( datetime, tableRGB )
    
    print( f"Execution Time: {time.time() - start_time:.3f}" )
    print( "\nDone\n" )
    return decrypt_vid

if __name__ == "__main__":
    year = gr.Slider(2000, 2024, 2023, step=1, label='year', info='iou threshold for filtering the annotations')
    month = gr.Slider(1, 12, 8, step=1, label='month', info='iou threshold for filtering the annotations')
    day = gr.Slider(1, 31, 1, step=1, label='day', info='iou threshold for filtering the annotations')
    hour = gr.Slider(1, 24, 17, step=1, label='hour', info='iou threshold for filtering the annotations')
    minute = gr.Slider(1, 60, 42, step=1, label='minute', info='iou threshold for filtering the annotations')
    
    global tableRGB
    tableRGB = loadTable()
    
    demo = gr.Interface( fn = decrypt, 
                         inputs = [year, month, day, hour, minute],                
                         outputs = "playable_video" ).launch()





























