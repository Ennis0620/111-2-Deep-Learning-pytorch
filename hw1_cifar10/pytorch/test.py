import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
import os,math,shutil,tqdm
import numpy as np
from dataset_loader import dataset_loader


def test(test_loader,
            model,
        ):
    '''
    測試
    '''
    test_acc = 0
    total_size = 0
    total_correct = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()    #預測要把model變成eval狀態
    with torch.no_grad():
        for img , label in tqdm.tqdm(test_loader):
            img = img.to(device)
            label = label.to(device)

            out = model(img)
            
            #累加每個batch的loss後續再除step數量
            
            testid_p = out.argmax(dim=1)   
            num_correct = (testid_p==label).sum().item() #該batch在train時預測成功的數量   
            total_correct += num_correct
            total_size += label.size(0)
            
        print('total_correct:',total_correct)    
        print('total_size:',total_size)

        test_acc = round((total_correct/total_size)*100,4)
    
    return test_acc   

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:',device)

    CURRENT_PATH = os.path.dirname(__file__)
    BEST_WEIGHT_NAME = f'epoch_78_trainLoss_0.6841_trainAcc_75.87_valLoss_0.6441_valAcc_77.9.pth'
    SAVE_MODELS_PATH = f'{CURRENT_PATH}/model_weight/5conv_2bn_mpl_before/{BEST_WEIGHT_NAME}'
    model = torch.load(SAVE_MODELS_PATH)
    print(model)
    print('Loading Weight:',BEST_WEIGHT_NAME)

    NEURAL_NETWORK = model 

    DATASET_NAME = 'CIFAR10'
    SPLIT_RATIO = 0.2
    SHUFFLE_DATASET = True
    BATCH_SIZE=128

    DL = dataset_loader(dataset_name=DATASET_NAME,
                        valid_split_ratio=SPLIT_RATIO,
                        shuffle=SHUFFLE_DATASET,
                        batch_size=BATCH_SIZE)
    test_dataloader = DL.test_dataloader(show_samples=False)
    test_acc = test(test_loader=test_dataloader,
                    model=NEURAL_NETWORK,)
    
    print('test Acc.{}%'.format(test_acc))