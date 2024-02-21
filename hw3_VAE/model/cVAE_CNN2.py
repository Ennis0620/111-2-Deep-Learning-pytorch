import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

class cVAE_CNN2(nn.Module):
    def __init__(self, input_image, latent_z, class_num, device) -> None:
        super(cVAE_CNN2, self).__init__()

        self.images_shape = input_image
        self.latent_z = latent_z
        self.class_num = class_num
        
        self.encoder = cVAE_Encoder(input_image, latent_z, class_num, device)
        self.decoder = cVAE_Decoder(input_image, latent_z, class_num, device)

    def forward(self, x, label):
        mu, logvar = self.encoder(x, label)
        z_latent = self.reparameterize(mu, logvar)
        reconstruction_img = self.decoder(z_latent, label)
        return reconstruction_img ,mu ,logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
class cVAE_Encoder(nn.Module):
    def __init__(self, input_image, latent_z, class_num, device):
        super(cVAE_Encoder,self).__init__()
        self.input_image = input_image
        self.latent_z = latent_z
        self.class_num = class_num   
        self.device = device 
        kernel_size = 3
        stride = 2
        #CIFAR10為RGB + 標籤1channel = 4 channel
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=32,
                               kernel_size=kernel_size,
                               stride=2,
                               padding=kernel_size//stride+1)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=2,
                               padding=kernel_size//stride+1)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=2,
                               padding=kernel_size//stride+1)

        self.liner1 = nn.Linear(128*6*6, 512)
        self.mu_liner =  nn.Linear(512, latent_z)
        self.logvar_liner =  nn.Linear(512, latent_z)

        self.selu = nn.SELU()                       #activation


    def forward(self, x, label):
        #將label reshpae成 H*W的形狀 將該channel全部變成label的數字
        #label = label.to(torch.int64)
        label = label.reshape(label.shape[0],1,1,1)
        reshape_label = torch.ones((x.shape[0],1,x.shape[2],x.shape[3])).to(self.device)
        label = (reshape_label*label)
        
        #cat 標籤 [2,3,32,32] + [2,1,32,32] => [2,4,32,32]
        x = torch.cat([x,label],dim=1)
        
        x = self.conv1(x)   
        x = self.selu(x)
        x = self.conv2(x)
        x = self.selu(x)
        x = self.conv3(x)
        x = self.selu(x)
        
        x = torch.flatten(x,start_dim=1)#影像flatten後輸入MLP
        x = self.liner1(x)

        z_mu = self.mu_liner(x)
        z_logvar = self.logvar_liner(x)
        
        return z_mu, z_logvar
    

class cVAE_Decoder(nn.Module):
    def __init__(self, input_image, latent_z, class_num, device):
        super(cVAE_Decoder,self).__init__()
        self.input_image = input_image
        self.latent_z = latent_z
        self.class_num = class_num 
        self.device = device
        kernel_size = 3
        stride = 2
        
        #decoder 一開始要加入label
        self.liner1 = nn.Linear(latent_z+class_num, 512)
        self.liner2 = nn.Linear(512, 128*6*6)
        
        self.conv1T = nn.ConvTranspose2d(in_channels=128,
                                         out_channels=64,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=kernel_size//stride+1)
        self.conv2T = nn.ConvTranspose2d(in_channels=64,
                                         out_channels=32,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=kernel_size//stride+1)
        self.conv3T = nn.ConvTranspose2d(in_channels=32,
                                         out_channels=3,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=kernel_size//stride+1)
        #最後把shape大小反捲積成原始大小
        self.conv4T = nn.ConvTranspose2d(in_channels=3,
                                         out_channels=3,
                                         kernel_size=6,
                                         )

        self.selu = nn.SELU()                       #activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, label):

        label_one_hot = self.one_hot(label, self.class_num).float().to(self.device)  # 使用 torch.eye() 进行 one-hot 编码
        
        #print(label_one_hot.size())
        z = torch.cat([z,label_one_hot],dim=1)     #將decoder的z + label標籤 cat再一起 : [2,64] + [2,10] => [2,74]
        
        z = self.liner1(z)
        z = self.liner2(z)  
        #reshape回影像[B,C,H,W] 準備做convTrans
        z = z.reshape(z.size(0),128,6,6)
        
        z = self.conv1T(z)
        z = self.selu(z)
        z = self.conv2T(z)
        z = self.selu(z)
        z = self.conv3T(z)
        z = self.selu(z)
        z = self.conv4T(z)
        #print(z.size())

        z = self.sigmoid(z)
        
        return z
    
    def one_hot(self, labels, num_classes):
        # label的shape是[batch,1(因為只有數字0~9,10類)] 要變成one hot 就要先創[0,0,0,0,0,0,0,0,0,0]
        one_hot = torch.zeros(labels.size(0), num_classes).to(self.device)
        #print(one_hot.size())
        #print(one_hot)
        # 使用 scatter_ 把label數值填上對應位置
        #ex: label=1  => [0,1,0,0,0,0,0,0,0,0] 
        #labels = labels.to(torch.int64)
        labels = torch.unsqueeze(labels,1).to(self.device)
        one_hot.scatter_(1, labels, 1)
        #print(one_hot)
        return one_hot

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    input_image = 3*32*32
    latent_z = 256
    class_num = 10
    model = cVAE_CNN2(input_image = input_image,
                    latent_z = latent_z,
                    class_num = class_num,
                    device = device).to(device)
    
    input_i = torch.empty(3, 3, 32, 32).to(device)
    label = torch.tensor([[1,],[2,],[3,]]).to(device)
    #print(input_i.size())
    #print(label.size())
    #print(model(input_i,label))    #要用整數檢查
    
    summary_in = torch.tensor([3,32,32])
    summary_label = torch.tensor([1,])
    print(summary(model, input_size=[summary_in, summary_label]))