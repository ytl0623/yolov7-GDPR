# yolov7/tools/instance.ipynb
# https://github.com/WongKinYiu/yolov7/blob/main/tools/instance.ipynb

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

def encrypt_image( image , mask_locs ):    
    # dateTime
    now = datetime.datetime.now(tz = datetime.timezone(datetime.timedelta(hours=8)))
    today = datetime.date.today()
    nowtime = now.strftime('%H-%M-%S')
    hour = nowtime.split('-')[0]
    minute = nowtime.split('-')[1]
    sec = nowtime.split('-')[2]
    
    # read image
    img = image
    
    # Path
    # root_dir = './'
    table_path = './Key_table'
    Table_Name = ['Table_R.mat', 'Table_G.mat', 'Table_B.mat']
    Alphabet = ['A','B','C']
    Table_R,Table_G,Table_B = [],[],[]
    save_path_img = os.path.join('./output/',str(today))
    save_path_mask = os.path.join('./mask_locs/',str(today))
    
    
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
    
    # 對像素質加密
    for mask_ind in range(len(mask_locs)):
        x , y = mask_locs[mask_ind][0] , mask_locs[mask_ind][1] # 第 mask_ind 個的mask location
        
        # 將像素值根據Table加密 dim : R,G,B
        img[x][y][0] = Table_R[img[x][y][0] % 256]
        img[x][y][1] = Table_G[img[x][y][1] % 256]
        img[x][y][2] = Table_B[img[x][y][2] % 256]
        
    # End of for loop
    
    # 檢查有無當下日期資料夾 (img)
    if not os.path.exists(save_path_img): 
        os.mkdir(save_path_img)
        
    # 檢查有無當下時間(hr)資料夾 (img)
    save_path_img = os.path.join(save_path_img,hour)
    if not os.path.exists(save_path_img): 
        os.mkdir(save_path_img)
        
    # 檢查有無當下時間(min)資料夾 (img)
    save_path_img = os.path.join(save_path_img,minute)
    if not os.path.exists(save_path_img): 
        os.mkdir(save_path_img)
    
    # 檢查有無當下日期資料夾 (mask locs)
    if not os.path.exists(save_path_mask): 
        os.mkdir(save_path_mask)
    
    # 檢查有無當下時間(hr)資料夾 (mask locs)
    save_path_mask = os.path.join(save_path_mask,hour)
    if not os.path.exists(save_path_mask): 
        os.mkdir(save_path_mask)
        
    # 檢查有無當下時間(min)資料夾 (mask locs)
    save_path_mask = os.path.join(save_path_mask,minute)
    if not os.path.exists(save_path_mask): 
        os.mkdir(save_path_mask)
    
    
    # Save Encrypted image and mask location info.
    # save_path = os.path.join(save_path_img,nowtime) # 路徑 './output/今天日期/現在時間.png'
    cv2.imwrite(os.path.join(save_path_img , sec) + '.png',img) # 存加密圖片
    # save_path = os.path.join(save_path_mask,nowtime) # 路徑 './mask_locs/今天日期/現在時間.npy'
    np.save(os.path.join(save_path_mask , sec),mask_locs) # 存當下 mask location 資訊
    
    return img # return Encrypted image

def decrypt_image( encrypted_image , datetime ): # 取 datetime 部分還沒完成
    # Read Encrypted Image
    img = encrypted_image
    
    # load mask location
    mask_locs = np.load(datetime + '.npy')
    
    # Path
    # root_dir = './'
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

    return img # return decrypted image

def find_indices(array):
    indices = np.argwhere(array == 1)
    return indices
    
def yolov7( image_path ) :
    start_time = time.time()

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

        #label = '%s %.3f' % (names[int(cls)], conf)
        #t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        #c2 = bbox[0] + t_size[0], bbox[1] - t_size[1] - 3
        #pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), c2, color, -1, cv2.LINE_AA)  # filled
        #pnimg = cv2.putText(pnimg, label, (bbox[0], bbox[1] - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        
    all_indices_array = np.delete( all_indices_array, 0, 0 )
    encrypt_img = encrypt_image( resized_original_img, all_indices_array )
    
    if isFall == True:
        cv2.putText(encrypt_img, f"Fall", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(encrypt_img, f"Safe", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #decrypt_img = decrypt_image( encrypt_img, "" )
    #cv2.imshow("decrypt",img)
    #cv2.waitKey(0)

    #print(type(pnimg[one_mask]))
    #print(pnimg[one_mask])

    #cv2.imwrite( "one_person_instance_seg.jpg", pnimg )
    #cv2.imshow("image",image_)
    #cv2.waitKey(0)
    #cv2.imshow("pnimg",pnimg)
    #cv2.waitKey(0)
    #image3 = cv2.subtract(image_, pnimg)
    #cv2.imshow("image3",image3)
    #cv2.waitKey(0)
    #cv2.imwrite( "one_person_body.jpg", image3 )

    print( f"Time: {time.time() - start_time:.3f} sec" )
    return encrypt_img

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    gr.Interface( fn = yolov7,
                  inputs = "image",
                  outputs = "image" ).launch()































