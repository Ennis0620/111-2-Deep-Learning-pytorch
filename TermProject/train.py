import os,shutil,tqdm
import cv2,re
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from early_stop import early_stop
from dataloader import train_data

current_path = f'H:\\NCNU\\class\\111-2_code\\2.DL\\111-2 DL HW\\TermProject'
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

train_df_grouped = train_df_grouped[(train_df_grouped['case'] != 7) | (train_df_grouped['day'] != 0)].reset_index(drop = True)
train_df_grouped = train_df_grouped[(train_df_grouped['case'] != 81) | (train_df_grouped['day'] != 30)].reset_index(drop = True)
train_df_grouped.to_csv('train_df_grouped.csv',index=False)

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

def dice_coeff_numpy(pred, target):
    smooth = 1.
    pred = pred
    target = target
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def dice_loss_numpy(pred, target, multiclass=False):
    if multiclass:
        loss = 0
        for c in range(target.shape[1]):
            dice = dice_coeff_numpy(pred[:, c, :, :], target[:, c, :, :])
            loss += (1. - dice)
        return loss / target.shape[1]
    else:
        dice = dice_coeff_numpy(pred, target)
        loss = 1. - dice
        return loss

import torch.nn as nn
from torch import optim
from model.model import U_net
import torch.nn.functional as F

def plot_statistics(train_loss,
                    train_performance_metric,
                    valid_loss,
                    valid_performance_metric,
                    performance_metric_name,
                    SAVE_MODELS_PATH):
    '''
    統計train、valid的loss、performance_metric
    '''
    fig, ax = plt.subplots()
    epcoh = [x for x in range(len(train_loss))]
    
    ax2 = ax.twinx()
    t_loss = ax.plot(train_loss,color='green',label='train_loss')
    v_loss = ax.plot(valid_loss,color='red',label='valid_loss')
    t_performance_metric = ax2.plot(train_performance_metric,color='#00FF55',label='train_acc')
    v_performance_metric = ax2.plot(valid_performance_metric,color='#FF5500',label='valid_acc')
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2.set_ylabel(f"{performance_metric_name}")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig(f'{SAVE_MODELS_PATH}/train_statistics',bbox_inches='tight')
    plt.figure()

def valid(model,
          validation_data_loader,
          criterion,
          n_classes):
    '''
    訓練時的驗證
    '''
    val_avg_loss = 0
    val_avg_dice = 0
    total_dice = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()    #預測要把model變成eval狀態
    with torch.no_grad():
        for img , masks in tqdm.tqdm(validation_data_loader):
            img = img.to(device)
            masks = masks.to(device)

            masks_pred = model(img)
            #多分類 和 單分類的metrics計算方式不同
            #loss = CE loss + dice loss
            loss = criterion(masks_pred, masks)
            #loss += dice_loss_numpy(masks_pred.detach().cpu().numpy(), masks.detach().cpu().numpy(), multiclass=True)
            
            #累加每個batch的loss後續再除step數量
            val_avg_loss += loss.item()

            #performance metrics: Dice coefficients 
            total_dice += dice_coeff_numpy(masks_pred.detach().cpu().numpy(), masks.detach().cpu().numpy())
            

        val_avg_loss = round(val_avg_loss/len(validation_data_loader),4)
        val_avg_dice = round(total_dice/len(validation_data_loader),4)

    return val_avg_loss,val_avg_dice
    
def train(model,
          training_data_loader,
          validation_data_loader,
          epochs,
          criterion,
          optimizer,
          n_classes,
          EARLY_STOP
          ):
    train_loss = []
    train_dice = []
    valid_loss = []
    valid_dice = []

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for num_epoch in range(epochs):
        train_avg_loss = 0                 #每個epoch的平均loss
        train_avg_dice = 0                 #每個epoch的平均dice
        for step, (img, masks) in tqdm.tqdm(enumerate(training_data_loader)):
            #確保每一batch都能進入model.train模式
            model.train()
            #放置gpu訓練
            img = img.to(device)
            masks = masks.to(device)
            #img經過nural network卷積後的預測(前向傳播),跟答案計算loss 
            masks_pred = model(img)

            #多分類 和 單分類的metrics計算方式不同
            #loss = CE loss + dice loss
            loss = criterion(masks_pred, masks)
            #loss += dice_loss_numpy(masks_pred.detach().cpu().numpy(), masks.detach().cpu().numpy(), multiclass=True)
            
            #performance metrics: Dice coefficients 
            train_avg_dice += dice_coeff_numpy(masks_pred.detach().cpu().numpy(), masks.detach().cpu().numpy())
            
            #優化器的gradient每次更新要記得初始化,否則會一直累積
            optimizer.zero_grad()
            #反向傳播偏微分,更新參數值
            loss.backward()
            #更新優化器
            optimizer.step()
            
            #累加每個batch的loss
            train_avg_loss += loss.item()

            
        #valid
        val_avg_loss,val_avg_dice = valid(
            validation_data_loader=validation_data_loader,
            model=model,
            criterion = criterion,
            n_classes = n_classes
        )    

        train_avg_loss = round(train_avg_loss/len(training_data_loader),4)
        train_avg_dice = round(train_avg_dice/len(training_data_loader),4)

        train_loss.append(train_avg_loss)
        train_dice.append(train_avg_dice)
        valid_loss.append(val_avg_loss)
        valid_dice.append(val_avg_dice)


        print('Epoch: {} | train_loss: {} | train_dice: {}% | val_loss: {} | val_dice: {}%'\
              .format(num_epoch, train_avg_loss,round(train_avg_dice*100,4),val_avg_loss,round(val_avg_dice*100,4)))
        
        #early stop
        performance_value = [num_epoch,
                             train_avg_loss,
                             round(train_avg_dice*100,4),
                             val_avg_loss,
                             round(val_avg_dice*100,4)]
        EARLY_STOP(val_avg_dice,
                   model=model,
                   performance_value = performance_value
                   )
        
        if EARLY_STOP.early_stop:
            print('Earlt stopping')
            break    


    return train_loss,train_dice,valid_loss,valid_dice 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_transform = transforms.Compose([
        #SquarePad(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
valid_transform = transforms.Compose([
        #SquarePad(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
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
batch_size = 16
whole_data = train_data(data_csv=train_df_grouped,
                           transforms=train_transform,
                           )
training_data_loader = DataLoader(dataset=whole_data,
                                  batch_size=batch_size,
                                  sampler = train_sampler)
validation_data_loader = DataLoader(dataset=whole_data,
                                  batch_size=batch_size,
                                  sampler = valid_sampler)

SAVE_MODELS_PATH = f'H:\\NCNU\\class\\111-2_code\\2.DL\\111-2 DL HW\\TermProject\\weights\\U_net'
EARLY_STOP = early_stop(save_path=SAVE_MODELS_PATH,
                        mode='max',
                        monitor='val_dice',
                        patience=10)
n_classes = 3
model = U_net(n_channels=1,
              n_classes=n_classes,
              bilinear=True).to(device)
#多分類:CrossEntropyLoss, 單分類:BCEWithLogitsLoss
criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
learning_rate = 0.001
weight_decay = 0.00001 #L2的regulization(防止overfitting)稱作weight_decay
momentum = 0.9
epochs = 100
'''
optimizer = optim.RMSprop(model.parameters(),
                          lr=learning_rate, 
                          weight_decay=weight_decay, 
                          momentum=momentum, foreach=True)
'''
optimizer = optim.Adam(model.parameters(),
                          lr=learning_rate, 
                          #weight_decay=weight_decay, 
                          )


train_loss,train_dice,valid_loss,valid_dice = train(
    training_data_loader=training_data_loader,
    validation_data_loader=validation_data_loader,
    model=model,
    epochs=epochs,
    criterion=criterion,
    optimizer=optimizer,
    n_classes=n_classes,
    EARLY_STOP = EARLY_STOP
)

plot_statistics(train_loss,train_dice,valid_loss,valid_dice,SAVE_MODELS_PATH,'dice')