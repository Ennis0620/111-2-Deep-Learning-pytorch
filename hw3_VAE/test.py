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
from utils.loss import loss_function
from utils.metric import calculate_fretchet,InceptionV3,inception_score,inception_score_torch
import numpy as np 
import matplotlib.pyplot as plt
from utils.early_stop import early_stop
from utils.metric import calculate_frechet_distance
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data

def test(test_loader,
            model,
            Inception_model
            ):
    '''
    訓練時的驗證
    '''
    test_avg_pfmetric = 0
    IS_score = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()    #預測要把model變成eval狀態
    with torch.no_grad():
        for num,(img , label) in tqdm.tqdm(enumerate(test_loader)):
            img = img.to(device)
            label = label.to(device)

            reconstruction_img ,mu ,logvar = model(img, label)
            reconstruction_img_reshape = torch.reshape(reconstruction_img,(len(label),3,32,32))#弄回2維圖片
            plot_images(reconstruction_img.detach().cpu().numpy().transpose(0,2,3,1),
                        label,
                        save_path=f'{num}'
                        )
            
            # FID = calculate_fretchet(img,reconstruction_img_reshape,Inception_model)
            # test_avg_pfmetric += FID
            score,_ = inception_score_torch(reconstruction_img_reshape, cuda=True, resize=True, splits=10)
            IS_score += score

        test_avg_pfmetric = round(test_avg_pfmetric/len(test_loader),4)
        IS_score = round(IS_score/len(test_loader),4)

    return test_avg_pfmetric,IS_score

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

    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)
        
    class LabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig, label, trans):
            self.orig = orig
            self.label = label
            self.trans = trans

        def __getitem__(self, index):
            img = self.trans(self.orig[index])
            return img,self.label[index]

        def __len__(self):
            return len(self.orig)

    test_transforms = transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Normalize((0,0,0), (1,1,1))
    ])

    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    input_image = 3*32*32
    latent_z = 196
    class_num = 10
    CURRENT_PATH = os.path.dirname(__file__)
    BATCH_SIZE = 512
    EPOCH = 10000
    LEARNING_RATE = 0.001
    '''
    SAVE_MODELS_PATH = f'{CURRENT_PATH}/weights/cVAE_MLP_loss_sum'
    WEIGHT = f'{SAVE_MODELS_PATH}/epoch_0_trainLoss_1008423.6734_trainFID_381.0294_valLoss_957025.6125_valFID_366.1835.pth'
    
    MODEL = cVAE_MLP(input_image=input_image,
                        latent_z=latent_z,
                        class_num=class_num,
                        device=device).to(device)
    '''
    SAVE_MODELS_PATH = f'{CURRENT_PATH}/weights/cVAE_CNN3_latent_z_196'
    WEIGHT = f'{SAVE_MODELS_PATH}/epoch_960_trainLoss_925363.8406_trainFID_0_valLoss_0_valFID_0.pth'
    MODEL = cVAE_CNN3(input_image=(3,32,32),
                        latent_z=latent_z,
                        class_num=class_num,
                        device=device).to(device)
     
    MODEL.load_state_dict(torch.load(WEIGHT))

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    Inception_model = InceptionV3([block_idx]).to(device)


    ##隨機產生 latent_z 並給予 condition
    for i in range(10):
        with torch.no_grad():
            c = torch.tensor([i]*1000).to(device)

            sample = torch.randn(1000, latent_z).to(device)
            #CNN
            sample = MODEL.decoder(sample, c).detach().cpu().numpy().transpose(0,2,3,1)

            d = LabelDataset(orig=sample,
                             label=c,
                             trans = test_transforms)
            d = Data.DataLoader(d,batch_size=500)

            test_pfmetric,IS_score = test(test_loader=d,      
                         model=MODEL,
                         Inception_model = Inception_model)
    
            print('FID:',test_pfmetric)  
            print('IS_score:',IS_score)  
            
            #MLP
            # sample = MODEL.decoder(sample, c)
            # sample = torch.reshape(sample,(len(c),3,32,32)).detach().cpu().numpy().transpose(0,2,3,1)
            
            # plot_images(images = sample,
            #             cls_true = c.cpu(),
            #             save_path = 'sample_' + str(i) + '.png')
            

    
    # data = dataset_loader(dataset_name="CIFAR10",
    #                       valid_split_ratio=0.05,
    #                       shuffle=True,
    #                       batch_size=BATCH_SIZE)
    
    # test_loader = data.test_dataloader(show_samples=False)

    # test_pfmetric,IS_score = test(test_loader=test_loader,      
    #                      model=MODEL,
    #                      Inception_model = Inception_model)
    
    # print('FID:',test_pfmetric)  
    # print('IS_score:',IS_score)  



    
        
    cifar = torchvision.datasets.CIFAR10(root='data/', 
                            download=True,
                            transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0,0,0), (1,1,1))
                             ]),
                             train=False,                      
    )
    
    
    #print (inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=32, resize=True, splits=10))
    
    from torchvision.utils import save_image



    