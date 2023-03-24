import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
import os,math,shutil,tqdm
import numpy as np
from dataset_loader import dataset_loader
from network import network
from early_stop import early_stop

def train(train_loader,
          valid_loader,
          model,
          batch_size,
          optimizer,
          loss_func,
          epoch,
          ):
    '''
    訓練
    '''
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    best_acc = 0
    best_loss = 0
    best_val_acc = 0
    best_val_loss = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for num_epoch in range(epoch):
        train_avg_loss = 0                #每個epoch的loss
        train_avg_acc = 0                 #每個epoch的acc
        total_acc = 0
        for step, (img, label) in tqdm.tqdm(enumerate(train_loader)):
            #確保每一batch都能進入model.train模式
            model.train()
            #放置gpu訓練
            img = img.to(device)
            label = label.to(device)
            #img經過nural network卷積後的預測(前向傳播),跟答案計算loss 
            out = model(img)
            loss = loss_func(out,label)
            #優化器的gradient每次更新要記得初始化,否則會一直累積
            optimizer.zero_grad()
            #反向傳播偏微分,更新參數值
            loss.backward()
            #更新優化器
            optimizer.step()

            #累加每個batch的loss後續再除step數量
            train_avg_loss += loss.item()
            
            #計算acc
            train_p = out.argmax(dim=1)                 #取得預測的最大值
            num_correct = (train_p==label).sum().item() #該batch在train時預測成功的數量
            batch_acc  = num_correct / label.size(0)
            total_acc += batch_acc


        val_avg_loss,val_avg_acc = valid(
            valid_loader=valid_dataloader,
            model=model,
            loss_func=loss_func
        )    
        
        train_avg_loss = round(train_avg_loss/len(train_loader),4)   #該epoch每個batch累加的loss平均
        train_avg_acc = round(total_acc/len(train_loader),4)         #該epoch的acc平均

        train_loss.append(train_avg_loss)
        train_acc.append(train_avg_acc)
        valid_loss.append(val_avg_loss)
        valid_acc.append(val_avg_acc)

        print('Epoch: {} | train_loss: {} | train_acc: {}% | val_loss: {} | val_acc: {}%'\
              .format(num_epoch, train_avg_loss,round(train_avg_acc*100,4),val_avg_loss,round(val_avg_acc*100,4)))
        
        #early stop
        performance_value = [num_epoch,
                             train_avg_loss,
                             round(train_avg_acc*100,4),
                             val_avg_loss,
                             round(val_avg_acc*100,4)]
        EARLY_STOP(val_avg_acc,
                   model=model,
                   performance_value = performance_value
                   )
        
        if EARLY_STOP.early_stop:
            print('Earlt stopping')
            break    


    return train_loss,train_acc,valid_loss,valid_acc 

def valid(valid_loader,
            model,
            loss_func,
            ):
    '''
    訓練時的驗證
    '''
    val_avg_loss = 0
    val_avg_acc = 0
    total_acc = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()    #預測要把model變成eval狀態
    with torch.no_grad():
        for img , label in valid_loader:
            img = img.to(device)
            label = label.to(device)

            out = model(img)
            loss = loss_func(out,label)
            #累加每個batch的loss後續再除step數量
            val_avg_loss += loss.item()

            valid_p = out.argmax(dim=1)   
            num_correct = (valid_p==label).sum().item() #該batch在train時預測成功的數量   
            batch_acc  = num_correct / label.size(0)
            total_acc += batch_acc
            

        val_avg_loss = round(val_avg_loss/len(valid_loader),4)
        val_avg_acc = round(total_acc/len(valid_loader),4)

    return val_avg_loss,val_avg_acc

def plot_statistics(train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc):
    '''
    統計train、valid的loss、acc
    '''
    
    t_loss = plt.plot(train_loss)
    t_acc = plt.plot(train_acc)
    v_loss = plt.plot(valid_loss)
    v_acc = plt.plot(valid_acc)
    plt.legend([t_loss,t_acc,v_loss,v_acc],
               labels=['train_loss',
                        'train_acc',
                        'valid_loss',
                        'valid_acc'])
    plt.savefig('train_statistics',bbox_inches='tight')
    plt.show()
    


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:',device)
    CURRENT_PATH = os.path.dirname(__file__)
    NEURAL_NETWORK = network(10).to(device=device)

    DATASET_NAME = 'CIFAR10'
    SPLIT_RATIO = 0.2
    SHUFFLE_DATASET = True
    BATCH_SIZE=512


    SAVE_MODELS_PATH = f'{CURRENT_PATH}/model_weight'
    LEARNING_RATE = 0.001 #lambda x: ((1.001 + math.cos(x * math.pi / EPOCH))) #* (1 - 0.1) + 0.1  # cosine
    EPOCH = 1000
    
    OPTIMIZER = torch.optim.Adam(NEURAL_NETWORK.parameters(), lr=LEARNING_RATE)
    #OPTIMIZER = torch.optim.SGD(NEURAL_NETWORK.parameters(), momentum=0.9, lr=LEARNING_RATE)
    #OPTIMIZER = torch.optim.RMSprop(NEURAL_NETWORK.parameters(), momentum=0.9, lr=LEARNING_RATE)
    

    LOSS = nn.CrossEntropyLoss()
    #LOSS = nn.MSELoss()

    EARLY_STOP = early_stop(save_path=SAVE_MODELS_PATH,
                            mode='max',
                            monitor='val_acc',
                            patience=10)

    DL = dataset_loader(dataset_name=DATASET_NAME,
                        valid_split_ratio=SPLIT_RATIO,
                        shuffle=SHUFFLE_DATASET,
                        batch_size=BATCH_SIZE)

    train_dataloader,valid_dataloader = DL.train_valid_dataloader(show_samples=False,aug=True)
    train_loss,train_acc,valid_loss,valid_acc = train(
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        model=NEURAL_NETWORK,
        batch_size=BATCH_SIZE,
        optimizer=OPTIMIZER,
        loss_func=LOSS,
        epoch=EPOCH
    )

    plot_statistics(train_loss,train_acc,valid_loss,valid_acc)