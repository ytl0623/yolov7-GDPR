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

def encrypt( image_path ) :
    print( "\nStart\n" )
    start_encrypt = time.time()

    with open('data/hyp.scratch.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    #weigths = torch.load('yolov7-seg.pt')
    weigths = torch.load('./models/yolov7-mask.pt')
    model = weigths['model']
    model = model.half().to(device)
    _ = model.eval()
    
    #image = cv2.imread(image_path)
    image = image_path
    image = letterbox(image, 640, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    image = image.half()
    
    output = model(image)
    
    inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], output['mask_iou'], output['bases'], output['sem']
    bases = torch.cat([bases, sem_output], dim=1)
    nb, _, height, width = image.shape
    names = model.names
    pooler_scale = model.pooler_scale
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)

    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)

    pred, pred_masks = output[0], output_mask[0]
    base = bases[0]
    bboxes = Boxes(pred[:, :4])
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
    pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    #nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)  # why change color?

    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(int)

    pnimg = nimg.copy()
    resized_original_img = nimg.copy()
    #cv2.imwrite( "one_person_resized.jpg", resized_original_img )  # save original resized photo

    all_indices_array = np.array([[0,0]])
    isFall = False
    fall_bbox = [] * 2
    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        if ( conf < 0.25 or cls != 0 ):  #inference model with desire class
            continue
            
        indices = find_indices( one_mask )
        indices_array = np.array(indices)  # collect all numpy array of person mask index
        
        # Concatenate the arrays along the rows
        all_indices_array = np.concatenate((all_indices_array, indices_array), axis=0) 
        np.save("./",all_indices_array)
        #color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        color = [0, 0, 0]  # black

        #pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        pnimg[one_mask] = np.array(color, dtype=np.uint8)
        
        #pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # safe or fall down
        #print(bbox[0], bbox[1], bbox[2], bbox[3], type(bbox[0]))
        if ( abs( bbox[2] - bbox[0] ) > abs( bbox[3] - bbox[1]) ):
            isFall = True
            fall_bbox.append( [bbox[0], bbox[1], bbox[2], bbox[3]] )
        
    all_indices_array = np.delete( all_indices_array, 0, 0 )
    encrypt_img = encrypt_image( resized_original_img, all_indices_array, tableRGB )
    
    if isFall == True:
        cv2.putText(encrypt_img, f"Fall", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        i = 0
        while i < len(fall_bbox):
                    cv2.rectangle(encrypt_img, (fall_bbox[i][0], fall_bbox[i][1]), (fall_bbox[i][2], fall_bbox[i][3]), (255, 0, 0), 2)
                    i += 1
    else:
        cv2.putText(encrypt_img, f"Safe", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print( f"Execution Time: {time.time() - start_encrypt:.3f}" )
    print( "\nDone\n" )
    return encrypt_img

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
    decrypt_img = decrypt_image( datetime, tableRGB )
    
    print( f"Execution Time: {time.time() - start_time:.3f}" )
    print( "\nDone\n" )
    return decrypt_img

def clear():
        return None, None

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global tableRGB
    tableRGB = loadTable() 
   
    css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"
    
    en_image_imput = gr.Image(label="Input", type='pil')
    en_image_output = gr.Image(label="Output", interactive=False, type='pil')
    
    de_image_output = gr.Image(label="Output", interactive=False, type='pil')
    
    with gr.Blocks( css = css, title = 'Encrypt Image and Decrypt Image' ) as demo :
        with gr.Tab( "Encrypt Image" ) :
            # Images
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    en_image_imput.render()

                with gr.Column(scale=1):
                    en_image_output.render()
                    
            # Submit & Clear
            with gr.Row():
                with gr.Column():
                    encrypt_image = gr.Button( "Encrypt Image", variant='primary' )
                    clear_image = gr.Button("Clear", variant="secondary")
            
        encrypt_image.click( fn = encrypt,
                             inputs = en_image_imput,
                             outputs = de_image_output )

        clear_image.click(clear, outputs=[en_image_imput, en_image_output])
                          
        with gr.Tab( "Decrypt Image" ) :
            # Images
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    year = gr.Slider(2000, 2024, 2023, step=1, label='year', info='iou threshold for filtering the annotations')
                    month = gr.Slider(1, 12, 8, step=1, label='month', info='iou threshold for filtering the annotations')
                    day = gr.Slider(1, 31, 1, step=1, label='day', info='iou threshold for filtering the annotations')
                    hour = gr.Slider(1, 24, 17, step=1, label='hour', info='iou threshold for filtering the annotations')
                    minute = gr.Slider(1, 60, 42, step=1, label='minute', info='iou threshold for filtering the annotations')

                with gr.Column(scale=1):
                    de_image_output.render()
                    
            # Submit & Clear
            with gr.Row():
                with gr.Column():
                    decrypt_image = gr.Button( "Decrypt Image", variant='primary' )
                    clear_image = gr.Button("Clear", variant="secondary")
              
        decrypt_image.click( fn = decrypt,
                             inputs = [year, month, day, hour, minute],
                             outputs = de_image_output )
                                 
        clear_image.click(clear, outputs=[de_image_output, de_image_output])
        
    demo.launch()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
