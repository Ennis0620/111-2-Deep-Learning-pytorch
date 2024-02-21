import os ,shutil, tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import dataset_loader
from model.cVAE_MLP import cVAE_MLP
from model.cVAE_MLP2 import cVAE_MLP2
from model.cVAE_CNN import cVAE_CNN
from model.cVAE_CNN2 import cVAE_CNN2
from model.cVAE_CNN3 import cVAE_CNN3
from utils.loss import loss_function,loss_function_CNN
from utils.metric import calculate_fretchet,InceptionV3
import numpy as np 
import matplotlib.pyplot as plt
from utils.early_stop import early_stop
from utils.metric import calculate_frechet_distance


def train(model,
          train_loader,
          valid_loader,
          optimizer,
          loss_func,
          epoch,
          scheduler=False,
          FID_model=False
          ):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_loss = []
    valid_loss = []
    trian_pfmetric = []
    valid_pfmetric = []
    for num_epoch in range(epoch):
        train_avg_loss = 0                #每個epoch的loss
        train_avg_pfmetric = 0            #每個epoch的performance metrics
        
        for step, (img, label) in tqdm.tqdm(enumerate(train_loader)):
            #確保每一batch都能進入model.train模式
            model.train()
            #放置gpu訓練
            img = img.to(device)
            label = label.to(device)
            
            #img經過nural network卷積後的預測(前向傳播),跟答案計算loss 
            reconstruction_img ,mu ,logvar = model(img, label)
            reconstruction_img_reshape = torch.reshape(reconstruction_img,(len(label),3,32,32))#弄回2維圖片
            
            #若用CNN回來就是2維
            loss = loss_func(reconstruction_img, img, mu, logvar)
            # loss = loss_func(reconstruction_img, img, mu, logvar)
            #優化器的gradient每次更新要記得初始化,否則會一直累積
            optimizer.zero_grad()
            #反向傳播偏微分,更新參數值
            loss.backward()
            #更新優化器
            optimizer.step()

            if step == 0:
                reconstruction_numpy = reconstruction_img_reshape.detach().cpu().numpy().transpose(0,2,3,1)
                plot_images(images=reconstruction_numpy,
                            cls_true=label,
                            save_path=f'{SAVE_IMG_NAME}/{num_epoch}.jpg')

            #累加每個batch的loss後續再除step數量
            train_avg_loss += loss.item()

            #FID = calculate_fretchet(img,reconstruction_img_reshape,FID_model)
            #train_avg_pfmetric += FID

        '''
        val_avg_loss,val_avg_pfmetric = valid(
            valid_loader=valid_loader,
            model=model,
            loss_func=loss_func,
            #FID_model = FID_model
        )
        '''   
        #更新learning rate
        if scheduler:
            scheduler.step()
        val_avg_loss = 0
        val_avg_pfmetric = 0
        

        train_avg_loss = round(train_avg_loss/len(train_loader),4)            #該epoch每個batch累加的loss平均
        #train_avg_pfmetric = round(train_avg_pfmetric/len(train_loader),4)    #該epoch的pfmetric平均

        # print('Epoch: {} | train_loss: {} | train_FID: {} | val_loss: {} | val_FID: {}'\
        #       .format(num_epoch, train_avg_loss,round(train_avg_pfmetric,4),val_avg_loss,round(val_avg_pfmetric,4)))
        
        print('Epoch: {} | train_loss: {}'\
              .format(num_epoch, train_avg_loss))

        #預防崩壞
        if train_avg_loss > 10e10 or train_avg_loss < 10e-10:
            return train_loss,trian_pfmetric,valid_loss,valid_pfmetric
        
        train_loss.append(train_avg_loss)
        valid_loss.append(val_avg_loss)
        trian_pfmetric.append(train_avg_pfmetric)
        valid_pfmetric.append(val_avg_pfmetric)

        performance_value = [num_epoch,
                             train_avg_loss,
                             round(train_avg_pfmetric,4),
                             val_avg_loss,
                             round(val_avg_pfmetric,4)]
        
        EARLY_STOP(train_avg_loss,
                   model=model,
                   performance_value = performance_value
                   )
        if EARLY_STOP.early_stop:
            print('Earlt stopping')
            break 
    # trian_pfmetric = 0
    # valid_pfmetric = 0
    
    return train_loss,trian_pfmetric,valid_loss,valid_pfmetric

def valid(valid_loader,
            model,
            loss_func,
            FID_model=False
            ):
    '''
    訓練時的驗證
    '''
    val_avg_loss = 0
    val_avg_pfmetric = 0

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()    #預測要把model變成eval狀態
    with torch.no_grad():
        for img , label in valid_loader:
            img = img.to(device)
            label = label.to(device)

            reconstruction_img ,mu ,logvar = model(img, label)
            reconstruction_img_reshape = torch.reshape(reconstruction_img,(len(label),3,32,32))#弄回2維圖片
            loss = loss_func(reconstruction_img, img, mu, logvar)
            #累加每個batch的loss後續再除step數量
            val_avg_loss += loss.item()

            #FID = calculate_fretchet(img,reconstruction_img_reshape,FID_model)
            #val_avg_pfmetric += FID

        val_avg_loss = round(val_avg_loss/len(valid_loader),4)
        #val_avg_pfmetric = round(val_avg_pfmetric/len(valid_loader),4)

    return val_avg_loss,val_avg_pfmetric

def plot_statistics(train_loss,
                    valid_loss,
                    train_performance,
                    valid_performance,
                    performance_name,
                    SAVE_MODELS_PATH):
    '''
    統計train、valid的loss、performance
    '''
    
    # t_loss = plt.plot(train_loss)
    # v_loss = plt.plot(valid_loss)
    
    # plt.legend([t_loss,v_loss],
    #            labels=['train_loss',
    #                     'valid_loss'])

    t_loss = plt.plot(train_loss)
    plt.legend([t_loss],
               labels=['train_loss'])


    plt.xlabel("epoch")
    plt.ylabel("loss")
    #plt.yscale('log')
    plt.savefig(f'{SAVE_MODELS_PATH}/train_loss',bbox_inches='tight')
    plt.figure()


    '''
    t_per = plt.plot(train_performance)
    v_per = plt.plot(valid_performance)

    plt.legend([t_per,v_per],
               labels=[f'train_{performance_name}',
                       f'valid_{performance_name}'])

    plt.xlabel("epoch")
    plt.ylabel(f"{performance_name}")
    plt.savefig(f'{SAVE_MODELS_PATH}/train_performance',bbox_inches='tight')
    plt.figure()
    '''

def plot_images(images, 
                cls_true, 
                cls_pred=None,
                save_path=None):
        """
        Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
        """

        fig, axes = plt.subplots(2,5)
        fig.set_figheight(10)
        fig.set_figwidth(50)
        class_index = [] 
        tmp_set = set()
        label_names = [
                        'airplane',
                        'automobile',
                        'bird',
                        'cat',
                        'deer',
                        'dog',
                        'frog',
                        'horse',
                        'ship',
                        'truck'
                    ]

        #取得1~10類 每類的圖
        for idx,i in enumerate(cls_true):
            if i in tmp_set:
                continue
            else:
                tmp_set.add(i)
                class_index.append(idx)
            if len(tmp_set)==10:
                break

        for i, (ax,idx) in enumerate(zip(axes.flat,class_index)):
            # plot img
            ax.imshow(images[idx, :, :, :],)

            # show true & predicted classes
            cls_true_name = label_names[cls_true[idx]]
            if cls_pred is None:
                xlabel = "{0} ({1})".format(cls_true_name, cls_true[idx])
            else:
                cls_pred_name = label_names[cls_pred[idx]]
                xlabel = "True: {0}, Pred: {1}".format(
                    cls_true_name, cls_pred_name
                )
            ax.set_xlabel(xlabel, fontsize = 30)
            ax.set_xticks([])
            ax.set_yticks([])
            
        #plt.show()  
        plt.savefig(f'{save_path}',bbox_inches='tight')
        plt.figure()  
        plt.cla()
        plt.close("all")
        




if __name__ == '__main__':
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    input_image = 3*32*32
    latent_z = 256
    class_num = 10
    CURRENT_PATH = os.path.dirname(__file__)
    SAVE_MODELS_PATH = f'{CURRENT_PATH}/weights/cVAE_CNN_latent_z_256'
    SAVE_IMG_NAME = f'{CURRENT_PATH}/weights/cVAE_CNN_train_img_latent_z_256'
    try:
        shutil.rmtree(SAVE_MODELS_PATH)
        shutil.rmtree(SAVE_IMG_NAME)
    except:
        pass
    os.makedirs(SAVE_MODELS_PATH)
    os.makedirs(SAVE_IMG_NAME)
    
        
    EARLY_STOP = early_stop(save_path=SAVE_MODELS_PATH,
                        mode='min',
                        monitor='train_loss',
                        patience=20)
    
    BATCH_SIZE = 512  
    EPOCH = 1000
    LEARNING_RATE = 0.0001
    '''
    MODEL = cVAE_MLP2(input_image=input_image,
                        latent_z=latent_z,
                        class_num=class_num,
                        device=device)
    
    '''
    MODEL = cVAE_CNN(input_image=(3,32,32),
                        latent_z=latent_z,
                        class_num=class_num,
                        device=device).to(device)
    
    #block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    #Inception_model = InceptionV3([block_idx]).to(device)

    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
    # SCHEDULER = torch.optim.lr_scheduler.CyclicLR(OPTIMIZER, base_lr=0.0001, 
    #                                           max_lr=0.0005,
    #                                           step_size_up=20,
    #                                           mode="exp_range",
    #                                           cycle_momentum=False)
    #LOSS = loss_function
    LOSS = loss_function_CNN
    
    data = dataset_loader(dataset_name="CIFAR10",
                          valid_split_ratio=0.0,
                          shuffle=True,
                          batch_size=BATCH_SIZE)
    
    train_loader,valid_loader = data.train_valid_dataloader(show_samples=False,
                                                                  aug=False)

    train_loss,train_pfmetric,valid_loss,valid_pfmetric = train(train_loader=train_loader,
          valid_loader=valid_loader,
          model=MODEL,
          optimizer=OPTIMIZER,
          loss_func=LOSS,
          epoch=EPOCH,)
          #scheduler = SCHEDULER)
          #FID_model = Inception_model)
    
    plot_statistics(train_loss,
                    valid_loss,
                    train_pfmetric,
                    valid_pfmetric,
                    performance_name='FID',
                    SAVE_MODELS_PATH=SAVE_MODELS_PATH
                    )