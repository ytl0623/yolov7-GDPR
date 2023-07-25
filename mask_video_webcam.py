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

def loadTable():
    table_path = './Key_table'
    Table_R,Table_G,Table_B = [],[],[]
    # Table_Name = ['Table_R.mat', 'Table_G.mat', 'Table_B.mat']
    # Alphabet = ['A','B','C']
    
    # load Table and Encrypt
    Path = os.path.join(table_path,'Table_R.mat') # Table Name
    Table_R = loadmat(Path) # load Table
    Table_R = Table_R['A'] # Table list
    Table_R = Table_R[1] # Take Encrypted pixel value
    
    Path = os.path.join(table_path,'Table_G.mat') # Table Name
    Table_G = loadmat(Path) # load Table
    Table_G = Table_G['B'] # Table list
    Table_G = Table_G[1] # Take Encrypted pixel value
    
    Path = os.path.join(table_path,'Table_B.mat') # Table Name
    Table_B = loadmat(Path) # load Table
    Table_B = Table_B['C'] # Table list
    Table_B = Table_B[1] # Take Encrypted pixel value
    
    return Table_R, Table_G, Table_B

def encrypt_image( image , mask_locs , T_R, T_G , T_B ):
    '''
    加密影像
    parameter : image : 要加密的影像 ; mask_locs : 需要加密的pixel位置 ; T_R,T_G,T_B : 要使用的加密對照表
    Output : img : 加密後的影像
    '''
    
    # dateTime
    now = datetime.datetime.now(tz = datetime.timezone(datetime.timedelta(hours=8)))
    today = datetime.date.today()
    nowtime = now.strftime('%H-%M-%S')
    hour = nowtime.split('-')[0]
    minute = nowtime.split('-')[1]
    sec = nowtime.split('-')[2]
    
    # read image
    img = image
    save_img_dir = os.path.join(r'C:\Users\User\Downloads\yolov7\output',str(today))
    save_mask_dir = os.path.join(r'C:\Users\User\Downloads\yolov7\mask_locs',str(today))
    
    # 對像素質加密
    for mask_ind in range(len(mask_locs)):
        x , y = mask_locs[mask_ind][0] , mask_locs[mask_ind][1] # 第 mask_ind 個的mask location
        
        # 將像素值根據Table加密 dim : R,G,B
        img[x][y][0] = T_R[img[x][y][0] % 256]
        img[x][y][1] = T_G[img[x][y][1] % 256]
        img[x][y][2] = T_B[img[x][y][2] % 256]
        
    # End of for loop
    
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
    # save_path = os.path.join(save_path_img,nowtime) # 路徑 './output/今天日期/現在時間.png'
    save_img_path = os.path.join(save_img_dir , sec)
    save_mask_path = os.path.join(save_mask_dir , sec)
    
    cv2.imwrite(save_img_path + '.jpg',img) # 存加密圖片  # .png 4x size
    # save_path = os.path.join(save_path_mask,nowtime) # 路徑 './mask_locs/今天日期/現在時間.npy'
    np.save(save_mask_path,mask_locs) # 存當下 mask location 資訊
    
    return img # return Encrypted image

def decrypt_image( encrypted_image , datetime , T_R , T_G , T_B ): # 取 datetime : yyyy-mm-dd (GUI input?)
    # Read Encrypted Image
    img = encrypted_image
    
    # load mask location
    mask_locs = np.load(datetime + '.npy')
    
    # Decrypted image
    for mask_ind in range(len(mask_locs)):
        
        # 第 mask_ind 個的mask location
        x,y = mask_locs[mask_ind][0],mask_locs[mask_ind][1]
        
        # 根據 Table 解密pixel
        img[x][y][0], img[x][y][1], img[x][y][2] = T_R.index(img[x][y][0]), T_G.index(img[x][y][1]), T_B.index(img[x][y][2])
    
    # End of for loop

    return img # return decrypted image

def find_indices(array):
    indices = np.argwhere(array == 1)
    return indices

def yolov7( image_path ):
    start_time = time.time()

    with open('data/hyp.scratch.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    #weigths = torch.load('yolov7-seg.pt')
    weigths = torch.load('./models/yolov7-mask.pt')
    model = weigths['model']
    model = model.float().to(device)
    _ = model.eval()
    
    #image = cv2.imread(image_path)
    image = image_path
    image = letterbox(image, 640, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    image = image.float()
    
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
    fall_bbox = []
    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        if ( conf < 0.25 or cls != 0 ):  #inference model with desire class
            continue
            
        indices = find_indices( one_mask )
        indices_array = np.array(indices)  # collect all numpy array of person mask index
        
        # Concatenate the arrays along the rows
        all_indices_array = np.concatenate((all_indices_array, indices_array), axis=0) 

        #color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        color = [0, 0, 0]  # black

        #pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        pnimg[one_mask] = np.array(color, dtype=np.uint8)
        
        #pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # safe or fall down
        #print(bbox[0], bbox[1], bbox[2], bbox[3], type(bbox[0]))
        if ( abs( bbox[2] - bbox[0] ) > abs( bbox[3] - bbox[1]) ):
            isFall = True
            fall_bbox.append( bbox[0] )
            fall_bbox.append( bbox[1] )
            fall_bbox.append( bbox[2] )
            fall_bbox.append( bbox[3] )
        
    all_indices_array = np.delete( all_indices_array, 0, 0 )
    encrypt_img = encrypt_image( resized_original_img, all_indices_array, tableRGB[0], tableRGB[1], tableRGB[2] )
    
    if isFall == True:
        cv2.putText(encrypt_img, f"Fall", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(encrypt_img, (fall_bbox[0], fall_bbox[1]), (fall_bbox[2], fall_bbox[3]), (255, 0, 0), 2)
    else:
        cv2.putText(encrypt_img, f"Safe", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print( f"Execution Time: {time.time() - start_time:.3f}" )
    return encrypt_img

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global tableRGB
    tableRGB = loadTable()

    gr.Interface( fn = yolov7,
                  inputs = gr.Image(source="webcam", streaming=True, label = "Input Video"),
                  outputs = "image",
                  live = True ).launch()


























