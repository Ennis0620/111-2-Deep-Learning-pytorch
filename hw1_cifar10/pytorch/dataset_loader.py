import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as Data
import matplotlib.pyplot as plt

class dataset_loader:
    def __init__(self,
                 dataset_name:str,
                 valid_split_ratio:float,
                 shuffle:bool,
                 batch_size:int,
                 ):
        super(dataset_loader, self).__init__()
        self.DATA_PATH = './data'
        self.RANDOM_SEED = 304
        self.dataset_name = dataset_name
        self.valid_split_ratio = valid_split_ratio
        self.SHUFFLE_DATASET = shuffle
        self.BATCH_SIZE = batch_size
        self.label_names = 0

    def plot_images(self,
                    images, 
                    cls_true, 
                    cls_pred=None):
        """
        Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
        """
        fig, axes = plt.subplots(3, 3)

        for i, ax in enumerate(axes.flat):
            # plot img
            ax.imshow(images[i, :, :, :], interpolation='spline16')

            # show true & predicted classes
            cls_true_name = self.label_names[cls_true[i]]
            if cls_pred is None:
                xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
            else:
                cls_pred_name = self.label_names[cls_pred[i]]
                xlabel = "True: {0}\nPred: {1}".format(
                    cls_true_name, cls_pred_name
                )
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()    

    def train_valid_dataloader(self,
                           show_samples:bool,
                           aug:bool):
        '''
        train、valid dataloader
        '''
        if aug:
            train_transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                torchvision.transforms.RandomRotation(15),
                                torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                                
                            ]) 
        else:
            train_transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                                
                            ]) 
        valid_transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                        ])

        if self.dataset_name == 'CIFAR10':
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.DATA_PATH,
                download=True,
                train=True,
                transform=train_transform
            )
            valid_dataset = torchvision.datasets.CIFAR10(
                root = self.DATA_PATH,
                download=True,
                train=True,
                transform=valid_transform
            )
            self.label_names = [
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
        
        num_train = len(train_dataset)  #train dataset筆數
        indices = list(range(num_train))#列出0~num_train長度的數字index
        split_data = int(np.floor(self.valid_split_ratio * num_train))

        #隨機打亂index
        if self.SHUFFLE_DATASET:
            np.random.shuffle(indices)
        
        #分配train,valid
        train_indices , valid_indices = indices[split_data:] , indices[:split_data]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        

        #dataloaer
        train_loader = Data.DataLoader(
                        dataset=train_dataset,
                        batch_size=self.BATCH_SIZE,
                        sampler=train_sampler,          
        )
        valid_loader = Data.DataLoader(
                        dataset=valid_dataset,
                        batch_size=self.BATCH_SIZE,
                        sampler=valid_sampler,          
        )
    
        if show_samples:
            sampler_loader = Data.DataLoader(
                dataset=train_dataset,
                batch_size=9,
                shuffle=self.SHUFFLE_DATASET
            )
            data_iter = iter(sampler_loader)
            imgs,labels = data_iter.next()
            Img = imgs.numpy().transpose([0,2,3,1])
            self.plot_images(Img,labels)
        
        return train_loader , valid_loader
    
    def test_dataloader(self,
                    show_samples:bool):
        '''
        test dataloader
        '''
        test_transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                        ]) 
        test_dataset = torchvision.datasets.CIFAR10(root=self.DATA_PATH,
                                                    train=False,
                                                    download=True,
                                                    transform=test_transform)

        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=self.SHUFFLE_DATASET,
        )
        if self.dataset_name == 'CIFAR10':
            self.label_names = [
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
        if show_samples:
            
            sampler_loader = Data.DataLoader(
                dataset=test_dataset,
                batch_size=9,
                shuffle=self.SHUFFLE_DATASET
            )
            data_iter = iter(sampler_loader)
            imgs,labels = data_iter.next()
            Img = imgs.numpy().transpose([0,2,3,1])
            self.plot_images(Img,labels)

        return test_loader
    
if __name__ == "__main__":
    DATASET_NAME = 'CIFAR10'
    SPLIT_RATIO = 0.2
    SHUFFLE_DATASET = True
    BATCH_SIZE=128

    DL = dataset_loader(dataset_name=DATASET_NAME,
                        valid_split_ratio=SPLIT_RATIO,
                        shuffle=SHUFFLE_DATASET,
                        batch_size=BATCH_SIZE
    )
    train_dataloader,valid_dataloader = DL.train_valid_dataloader(show_samples=False)
    