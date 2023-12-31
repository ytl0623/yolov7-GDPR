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

from send_email import send_email

from multiprocessing import Process

def encrypt( video_path ):
    print( "\nStart\n" )
    start_encrypt = time.time()
    
    with open('data/hyp.scratch.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    #weigths = torch.load( model_path )  # dont forget modify function parameter
    weigths = torch.load( "./models/yolov7-mask.pt" )
    model = weigths['model']
    model = model.half().to(device)
    _ = model.eval()

    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')

    # Get the frame width and height.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    #print(frame_width)  # letterbox() should match 320*N

    # Pass the first frame through `letterbox` function to get the resized image,
    # to be used for `VideoWriter` dimensions. Resize by larger side.
    vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]

    save_name = f"{video_path.split('/')[-1].split('.')[0]}"

    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(f"{save_name}_encrypt.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (resize_width, resize_height))
    """
    out2 = cv2.VideoWriter(f"{save_name}_decrypt.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (resize_width, resize_height))"""

    frame_count = 0  # To count total frames.
    total_fps = 0  # To get the final frames per second.
    
    while (cap.isOpened):
        #start_image = time.time()
        # Capture each frame of the video.
        ret, frame = cap.read()

        if ret :
            orig_image = frame
            image = orig_image.copy()
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]  # default: 640, 1280
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            image = image.to(device)
            image = image.half()
            
            # Get the start time.
            start_time = time.time()

            with torch.no_grad():
                output = model(image)  # !!!

            # Get the end time.
            end_time = time.time()

            # Get the fps.
            fps = 1 / (end_time - start_time)

            # Add fps to total fps.
            total_fps += fps

            # Increment frame count.
            frame_count += 1

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
            nbboxes = bboxes.tensor.detach().cpu().numpy().astype(int)
            pnimg = nimg.copy()
            resized_original_img = nimg.copy()  # save original resized dphoto

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
                
                color = [0, 0, 0]  # black
                pnimg[one_mask] = np.array(color, dtype=np.uint8)
                
                # safe or fall down
                #print(bbox[0], bbox[1], bbox[2], bbox[3], type(bbox[0]))
                if ( abs( bbox[2] - bbox[0] ) > abs( bbox[3] - bbox[1]) ):
                    isFall = True
                    fall_bbox.append( [bbox[0], bbox[1], bbox[2], bbox[3]] )

            #out.write(pnimg)

            #image3 = cv2.subtract(image_, pnimg)
            #out2.write(image3)
            
            # Encrypt and Decrypt
            all_indices_array = np.delete( all_indices_array, 0, 0 )
            encrypt_img = encrypt_image( resized_original_img, all_indices_array, tableRGB )  # avg. 1 sec/image
            
            # Write the FPS on the current frame.
            cv2.putText(encrypt_img, f"FPS: {fps:.3f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if isFall == True:
                p1 = Process( target = send_email )
                p1.start()
                cv2.putText(encrypt_img, f"Fall", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                i = 0
                while i < len(fall_bbox):
                    cv2.rectangle(encrypt_img, (fall_bbox[i][0], fall_bbox[i][1]), (fall_bbox[i][2], fall_bbox[i][3]), (0, 0, 255), 2)
                    i += 1
            else:
                cv2.putText(encrypt_img, f"Safe", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(encrypt_img)
            
            #decrypt_img = decrypt_image( encrypt_img, "" )
            #out2.write(decrypt_img)
        else:
            break

        #print(f"Time/Image: {(time.time() - start_image):.3f}")  # 0.1 sec/image

    # Release VideoCapture().
    cap.release()

    # Close all frames and video windows.
    cv2.destroyAllWindows()

    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    
    print(f"Average FPS: {avg_fps:.3f}")
    print( f"Execution Time: {time.time() - start_encrypt:.3f}" )
    print( "\nDone\n" )
    return f"{save_name}_encrypt.mp4"

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global tableRGB
    tableRGB = loadTable()

    gr.Interface( fn = encrypt,
                  inputs = "video",
                  outputs = "playable_video",
                  examples = ["./result/fall.mp4"], 
                  cache_examples = True ).launch()
"""
#python mask_video.py -d 0 -i ./result/2.mp4 -m ./models/yolov7-mask.pt
if __name__ == '__main__':
    start_main = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device')
    parser.add_argument('-i', '--input_path')
    parser.add_argument('-m', '--model_path')
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    global tableRGB
    tableRGB = loadTable()
    
    yolov7( args.input_path, args.model_path )

    print(f"Execution Time: {(time.time() - start_main):.3f}")"""





























