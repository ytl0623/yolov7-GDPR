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

from function import encrypt_image, decrypt_image, loadTable, find_indices

def decrypt( year, month, day, hour, minute ) :
    datetime = "./output/{}-0{}-{}/{}/{}/".format( year, month, day, hour, minute )
    return decrypt_image( datetime, tableRGB )

if __name__ == "__main__":
    year = gr.Slider(2000, 2024, 2023, step=1, label='year', info='iou threshold for filtering the annotations')
    month = gr.Slider(1, 12, 7, step=1, label='month', info='iou threshold for filtering the annotations')
    day = gr.Slider(1, 31, 26, step=1, label='day', info='iou threshold for filtering the annotations')
    hour = gr.Slider(1, 24, 17, step=1, label='hour', info='iou threshold for filtering the annotations')
    minute = gr.Slider(1, 60, 53, step=1, label='minute', info='iou threshold for filtering the annotations')
    
    global tableRGB
    tableRGB = loadTable()
    
    # It's not suitable for set example.
    # problem: 1. need "07" not "7" 2. need filename not [SECOND].jpg, [SECOND].npy
    # decrypt video needs decrypt whole directory
    demo = gr.Interface( fn = decrypt, 
                         inputs = [year, month, day, hour, minute],                
                         outputs = "image" ).launch()





























