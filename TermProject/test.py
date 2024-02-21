import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision
#import torchvision.transforms.functional as F
import PIL.Image as Image
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import tqdm, os
import torch.nn as nn
from torch import optim
from model.model import U_net,SE_U_net,CBAM_U_net,CBAM_U_net_only_scout,SE_U_net_only_scout,SE_U_net_Encoder_CBAM_U_net_scout
import torch.nn.functional as F

class train_data(Dataset):
    def __init__(self,
                 data_csv,
                 transforms:False,):
        super(train_data,self).__init__()
        self.data_csv = data_csv
        self.classes = ['large_bowel','small_bowel','stomach']
        self.transform = transforms
        self.imgs = []
        self.masks = []
        for i in tqdm.tqdm(range(len(data_csv))):
            large_bowel_path = str(self.data_csv.iloc[i, :].large_bowel)
            small_bowel_path = str(self.data_csv.iloc[i, :].small_bowel)
            stomach_path = str(self.data_csv.iloc[i, :].stomach)
            img_path = str(self.data_csv.iloc[i, :].full_path)
            imgs_list = self.decode2Img(img_path=img_path,
                                    mask_path=[large_bowel_path,small_bowel_path,stomach_path])
            img = imgs_list[0]
            target_masks = imgs_list[1]
            
            #transpose numpy(C,H,W) 轉換 原始格式 (H,W,C)
            img = img.transpose((1,2,0))
            target_masks = target_masks.transpose((1,2,0))

            '''
            img_ = cv2.imread(img_path,0)
            h,w = img.shape[0],img.shape[1]
            cv2.imwrite(f'H:/NCNU/class/111-2_code/2.DL/111-2 DL HW/TermProject/mask/{i}_{w}_{h}.png',target_masks)
            '''

            #cv2.imshow('t',target_masks)
            #cv2.waitKey(0)
            
            self.imgs.append(img)
            self.masks.append(target_masks)
            
        
    def __getitem__(self,index):
        #torch (B,C,H,W)
        img = self.transform(self.imgs[index])
        mask = self.transform(self.masks[index])

        return img, mask 
    
    def __len__(self):
        
        return len(self.data_csv) 
    
    def RLE_encode(self,img):
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = img.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)


    def RLE_decoder(self, rle, shape , fill=1):
        '''
        PIL為(W,H)
        run-length encoding將二進位圖像中的像素值序列壓縮的技術,醫學影像中,RLE通常用於編碼二進位,節省空間並方便傳輸
        '''
        height, width = shape
        s = rle.split()
        start, length = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        start -= 1
        mask = np.zeros(height*width, dtype=np.uint8)
        for i, l in zip(start, length):
            mask[i:i+l] = fill
        mask = mask.reshape(height,width)
        mask = np.ascontiguousarray(mask)#.reshape(shape)
        return mask 
    
    def decode2Img(self,
                   img_path,
                   mask_path):
        '''
        @input
        `img_path`:給原始圖片路徑, data full_path欄位
        `mask_path`:依序給mask path的路徑[large_bowel, small_bowel, stomach]
        @return 
        `return_img_mask`:[img, large_bowel, small_bowel, stomach]
        '''
        img = cv2.imread(img_path,0)
        h,w = img.shape[0],img.shape[1]

        return_img_mask = []
        #torch shape:(C,H,W)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
            return_img_mask.append(img)
        
        concate_mask = []
        for c in mask_path:
            #print(c+'\n')
            trans2mask = self.RLE_decoder(c, (h,w))*255
            #trans2mask:shpae 為(h, w)要轉成(1, h, w)
            trans2mask = trans2mask[np.newaxis, ...]
            concate_mask.append(trans2mask)
        #將(1,h,w),(1,h,w),(1,h,w)組成(3,h,w)
        concate_mask = np.concatenate(concate_mask,axis=0)
        return_img_mask.append(concate_mask)
        
        return return_img_mask
    
class SquarePad:
	'''
	方形填充到長寬一樣大小再來resize
    '''
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = [hp, vp, hp, vp]
		return torchvision.transforms.functional.pad(image, padding, 0, 'constant')




CURRENT_PATH = os.path.dirname(__file__)
train_df_grouped = pd.read_csv(f'{CURRENT_PATH}/train_df_grouped.csv')
train_df_grouped = train_df_grouped.fillna('')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  
    ])
valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  
    ])

valid_split = 0.2
# random seed
random_seed = 1234
dataset_size = len(train_df_grouped[:])
indices = list(range(dataset_size))
split = int(np.floor(valid_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]

training_data = train_df_grouped.iloc[train_indices].reset_index()
validation_data = train_df_grouped.iloc[valid_indices].reset_index()


training_data = train_data(data_csv=training_data,
                        transforms=train_transform,
                        )

validation_data = train_data(data_csv=validation_data,
                        transforms=valid_transform,
                        )


batch_size = 16
training_data_loader = DataLoader(dataset=training_data,
                                  batch_size=batch_size,)
validation_data_loader = DataLoader(dataset=validation_data,
                                  batch_size=batch_size,)


n_classes = 3

# SAVE_MODELS_PATH = f'H:\\NCNU\\class\\111-2_code\\2.DL\\111-2 DL HW\\TermProject\\weights\\U_net_CyclicLR_dice_loss'
# SAVE_MODELS_PATH = f'H:\\NCNU\\class\\111-2_code\\2.DL\\111-2 DL HW\\TermProject\\weights\\SE_U_net_CyclicLR_dice_loss'
# SAVE_MODELS_PATH = f'H:\\NCNU\\class\\111-2_code\\2.DL\\111-2 DL HW\\TermProject\\weights\\CBMA_U_net_CyclicLR_dice_loss'
#SAVE_MODELS_PATH = f'H:\\NCNU\\class\\111-2_code\\2.DL\\111-2 DL HW\\TermProject\\weights\\CBAM_U_net_only_scout_CyclicLR_dice_loss'
# SAVE_MODELS_PATH = f'H:\\NCNU\\class\\111-2_code\\2.DL\\111-2 DL HW\\TermProject\\weights\\SE_U_net_only_scout_CyclicLR_dice_loss'
# SAVE_MODELS_PATH = f'H:\\NCNU\\class\\111-2_code\\2.DL\\111-2 DL HW\\TermProject\\weights\\SE_U_net_Encoder_CBAM_U_net_scout_CyclicLR_dice_loss'
SAVE_MODELS_PATH = f'H:\\NCNU\\class\\111-2_code\\2.DL\\111-2 DL HW\\TermProject\\weights\\SE_U_net_transpose_conv_dice_loss'

# model = SE_U_net_Encoder_CBAM_U_net_scout(n_channels=1,
#                 n_classes=n_classes,
#                 mode='avg_pool',
#                 bilinear=True).to(device)
# model = SE_U_net(n_channels=1,
#                 n_classes=n_classes,
#                 mode='avg_pool',
#                 bilinear=True).to(device)
# model = U_net(n_channels=1,
#                 n_classes=n_classes,
#                 bilinear=True).to(device)

# #

# model = SE_U_net_only_scout(n_channels=1,
#                 n_classes=n_classes,
#                 mode='avg_pool',
#                 bilinear=True).to(device)

model = SE_U_net(n_channels=1,
                n_classes=n_classes,
                mode='avg_pool',
                bilinear=False).to(device)

# model = U_net(n_channels=1,
#                 n_classes=n_classes,
#                 bilinear=True).to(device)

# model = CBAM_U_net(n_channels=1,
#                 n_classes=n_classes,
#                 bilinear=True).to(device)

# model.load_state_dict(torch.load(f'{SAVE_MODELS_PATH}/epoch_14_trainLoss_0.1281_trainDICE_86.9_valLoss_0.1383_valDICE_86.49.pth'))#U-net
# model.load_state_dict(torch.load(f'{SAVE_MODELS_PATH}/epoch_75_trainLoss_0.0958_trainDICE_91.42_valLoss_0.109_valDICE_90.07.pth'))#SE U-net
# model.load_state_dict(torch.load(f'{SAVE_MODELS_PATH}/epoch_9_trainLoss_0.3581_trainDICE_45.66_valLoss_0.3772_valDICE_47.02.pth'))#CBAM U-net
# model.load_state_dict(torch.load(f'{SAVE_MODELS_PATH}/epoch_38_trainLoss_0.102_trainDICE_89.72_valLoss_0.1178_valDICE_88.16.pth'))#CBAM shortcut U-net
# model.load_state_dict(torch.load(f'{SAVE_MODELS_PATH}/epoch_21_trainLoss_0.1057_trainDICE_90.5_valLoss_0.1174_valDICE_89.28.pth'))#SE shortcut U-net
# model.load_state_dict(torch.load(f'{SAVE_MODELS_PATH}/epoch_20_trainLoss_0.1088_trainDICE_90.1_valLoss_0.1188_valDICE_88.83.pth'))#SE encdoe CBAM decode U-net
model.load_state_dict(torch.load(f'{SAVE_MODELS_PATH}/epoch_101_trainLoss_0.0881_trainDICE_92.44_valLoss_0.1028_valDICE_90.73.pth'))#SE  U-net transpose


dd = iter(validation_data_loader)
valid_img,valid_mask = next(dd)
print(valid_img.size())
print(valid_mask.size())

model.eval()    #預測要把model變成eval狀態
mask_pred = model(valid_img.to(device))
mask_pred = torch.sigmoid(mask_pred)
print(torch.max(mask_pred))

mask_pred_numpy = mask_pred.detach().cpu().numpy()
valid_img_numpy = valid_img.detach().cpu().numpy()
valid_mask_numpy = valid_mask.detach().cpu().numpy()

# idx = 3
idx = 14
img_1 = valid_img_numpy[idx].transpose(1,2,0)

img_1 = img_1.squeeze(2)
img_1 = cv2.cvtColor(img_1,cv2.COLOR_GRAY2BGR)
print('img_1:',img_1.shape)
valid_mask_1 = valid_mask_numpy[idx].transpose(1,2,0)
print('valid_mask_1',valid_mask_1 .shape)
mask_1 = mask_pred_numpy[idx].transpose(1,2,0)
print('mask_1',mask_1.shape)

import cv2
mask_1 = (mask_1//np.max(mask_1))*255

_,img_th_c1 = cv2.threshold(mask_1[:,:,0],127,255,cv2.THRESH_BINARY)
_,img_th_c2 = cv2.threshold(mask_1[:,:,1],127,255,cv2.THRESH_BINARY)
_,img_th_c3 = cv2.threshold(mask_1[:,:,2],127,255,cv2.THRESH_BINARY)

img_th_c1 = cv2.cvtColor(img_th_c1,cv2.COLOR_GRAY2BGR)
print(img_th_c1.shape)
img_th_c2 = cv2.cvtColor(img_th_c2,cv2.COLOR_GRAY2BGR)
img_th_c3 = cv2.cvtColor(img_th_c3,cv2.COLOR_GRAY2BGR)

diver = np.ones_like(mask_1)[:,:5,:]
print(diver.shape)
con = np.concatenate([img_1,diver,
                      img_th_c1,diver,
                      img_th_c2,diver,
                      img_th_c3,diver,
                      mask_1,diver,
                      valid_mask_1,diver],axis=1)

cv2.imshow('con',con)
cv2.waitKey(0)
cv2.destroyAllWindows()
