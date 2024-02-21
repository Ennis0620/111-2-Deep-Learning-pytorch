import os,shutil,tqdm
import cv2,re
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

current_path = os.path.dirname(__file__)
traindata_path = f'{current_path}/traindata'
train_csv_path = f'{current_path}/train.csv'
train = pd.read_csv(train_csv_path)

#過濾掉segmentation為nan欄位
train = train.dropna(subset=['segmentation'])
train = train.reset_index(drop=True)

#取出目前要segmentation的3個部位
classes = train.loc[:, 'class'].unique().tolist()
print(classes)

#依照把3行欄位('stomach', 'large_bowel', 'small_bowel')化減成一行(因3segmentation部位是分別標記的,所以一張影像有3個欄位)
train_df_grouped = train.copy()
train_df_grouped.set_index('id', inplace = True)
seg_list = []
for cl in classes:
    seg = train_df_grouped[train_df_grouped['class'] == cl]['segmentation']
    seg.name = cl
    seg_list.append(seg)
    
train_df_grouped = pd.concat(seg_list, axis=1).reset_index()
train_df_grouped.fillna('', inplace = True)
print(train_df_grouped.head())


def get_case_day_slice(x):
    #--------------------------------------------------------------------------------------
    # function that parses a string (full_path or image_id)
    # and returns case, day, slice_ 
    #--------------------------------------------------------------------------------------
    case = re.search('case[0-9]+', x).group()[len('case'):]
    day = re.search('day[0-9]+', x).group()[len('day'):]
    slice_ = re.search('slice_[0-9]+', x).group()[len('slice_'):]
    return case, day, slice_


def process_df(df, path):
    #--------------------------------------------------------------------------------------
    # add new columns:
    # ['case', 'day', 'slice_']
    # and
    # 'full_path'
    #--------------------------------------------------------------------------------------
    case_day_slice = ['case', 'day', 'slice_']
    df = df.copy()
    df.loc[:, case_day_slice] = df.id.apply(get_case_day_slice).to_list()
    
    # get list of all images 
    all_images = glob(os.path.join(path, "**", "*.png"), recursive = True)
    img_df = pd.DataFrame(all_images, columns = ['full_path'])
    img_df.loc[:, case_day_slice] = img_df.full_path.apply(get_case_day_slice).to_list()
    
    return df.merge(img_df, on = case_day_slice, how = 'left')

train_df_grouped = process_df(train_df_grouped, traindata_path)
print(train_df_grouped.head())
print(len(train_df_grouped))

train_df_grouped = train_df_grouped[(train_df_grouped['case'] != 7) | (train_df_grouped['day'] != 0)].reset_index(drop = True)
train_df_grouped = train_df_grouped[(train_df_grouped['case'] != 81) | (train_df_grouped['day'] != 30)].reset_index(drop = True)

def RLE_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def RLE_decoder(rle,shape):
    '''
    run-length encoding將二進位圖像中的像素值序列壓縮的技術,醫學影像中,RLE通常用於編碼二進位,節省空間並方便傳輸
    '''
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

num = 5
segmentation_df_example = train_df_grouped[train_df_grouped.large_bowel != ''].sample(num)
segmentation_df_example

fig, ax = plt.subplots(num, 3, figsize=(18, 8*num))
for i in range(num):
    record = segmentation_df_example.iloc[i, :]
    
    #img = mpimg.imread(record.full_path, format = 'png')
    img = cv2.imread(record.full_path,cv2.IMREAD_GRAYSCALE)
    ax[i, 0].imshow(img)
    ax[i, 0].set_title(record.id)
    cv2.imwrite(f'origin.png',img)
    mask = np.zeros(img.shape)
    for j, cl in enumerate(classes):
        mm = RLE_decoder(record[cl], img.shape)*255
        cv2.imwrite(f'{cl}.png',mm)
        
        mask += RLE_decoder(record[cl], img.shape)*(j + 1) / 4 * np.max(img)
    cv2.imwrite(f'merge.png',mask)
    
    plt.savefig(f'{current_path}/mask.png')
    ax[i, 1].imshow(mask)
    ax[i, 2].imshow(img + mask)





'''
for root, dirs, files in os.walk(traindata_path):
    if files != []:     #當有走到存在圖片的資料夾才取圖
        root_b = root.split('scans')[0]
        #找到normalize的資料夾
        if root.split('\\')[-1]=='normalize':
            for img_name in tqdm.tqdm(files):
                print(img_name.split('_'))
                
                normalize_png = f'{root_b}/normalize'
                img = cv2.imread(f'{normalize_png}/{img_name}',cv2.IMREAD_GRAYSCALE)
                h,w = img.shape
'''