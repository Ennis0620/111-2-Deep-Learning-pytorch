import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision
#import torchvision.transforms.functional as F
import PIL.Image as Image
import numpy as np
import pandas as pd
import cv2
import tqdm


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
            #transpose 回PIL格式 (w,h,c)
            img = img.transpose((1,2,0))
            target_masks = target_masks.transpose((1,2,0))
            
            self.imgs.append(img)
            self.masks.append(target_masks)
        

    def __getitem__(self,index):
        '''
        large_bowel_path = str(self.data_csv.iloc[index, :].large_bowel)
        small_bowel_path = str(self.data_csv.iloc[index, :].small_bowel)
        stomach_path = str(self.data_csv.iloc[index, :].stomach)
        img_path = str(self.data_csv.iloc[index, :].full_path)
        

        imgs_list = self.decode2Img(img_path=img_path,
                                    mask_path=[large_bowel_path,small_bowel_path,stomach_path])
        
        #cv2.imwrite('test.png',np.transpose(imgs_list[1]),(2,1,0))

        img = imgs_list[0]
        target_masks = imgs_list[1]

        #transpose 回PIL格式 (w,h,c)
        img = img.transpose((1,2,0))
        target_masks = target_masks.transpose((1,2,0))
        
        if self.transform:
            img = self.transform(img)
            target_masks = self.transform(target_masks)
            #img = torch.from_numpy(img).float()
            #target_masks = torch.from_numpy(target_masks).float()
            return img, target_masks
        else:
            img = torch.from_numpy(img).float()
            target_masks = torch.from_numpy(target_masks).float()
        
        return img, target_masks
        '''

        return self.transform(self.imgs[index]), self.transform(self.masks[index])
    
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

    def RLE_decoder(self,
                    rle,shape):
        '''
        run-length encoding將二進位圖像中的像素值序列壓縮的技術,醫學影像中,RLE通常用於編碼二進位,節省空間並方便傳輸
        '''
        if rle=='nan':
            img = np.zeros((shape[0],shape[1]), dtype=np.uint8)
            return img 
        
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[::2], s[1::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)
    
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
        pil_img = Image.open(img_path).convert('L')
        w,h = pil_img.size
        img = np.asarray(pil_img)
        
        return_img_mask = []
        #torch shape:(C,H,W)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
            return_img_mask.append(img)
        
        concate_mask = []
        for c in mask_path:
            #print(c+'\n')
            #trans2mask:shpae 為(h, w)要轉成(1, h, w)
            trans2mask = self.RLE_decoder(c, (w,h))*255
            trans2mask = trans2mask[np.newaxis, ...]
            #print(trans2mask.shape)
            concate_mask.append(trans2mask)
        #將(1,h,w),(1,h,w),(1,h,w)組成(3,h,w)
        concate_mask = np.concatenate(concate_mask,axis=0)
        #print(concate_mask.shape)
        return_img_mask.append(concate_mask)
        
        return return_img_mask

train_transform = transforms.Compose([
        #SquarePad(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])

train_df_grouped = pd.read_csv('./train_df_grouped.csv')
valid_split = 0.2
# random seed
random_seed = 1234
dataset_size = len(train_df_grouped)
indices = list(range(dataset_size))
split = int(np.floor(valid_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]

# data samplers and loaders
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

training_data = train_data(data_csv=train_df_grouped,
                           transforms=train_transform,
                           )

training_data_loader = DataLoader(dataset=training_data,
                                  batch_size=3,
                                  sampler = train_sampler)

dataiter = iter(training_data_loader)
data = next(dataiter)
img, masks = data
print(img.shape)
print(masks.shape)