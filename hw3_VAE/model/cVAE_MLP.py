import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

class cVAE_MLP(nn.Module):
    def __init__(self, input_image, latent_z, class_num, device) -> None:
        super(cVAE_MLP, self).__init__()

        self.images_shape = input_image
        self.latent_z = latent_z
        self.class_num = class_num
        self.device = device
        
        self.encoder = cVAE_Encoder(input_image, latent_z, class_num, device).to(device)
        self.decoder = cVAE_Decoder(input_image, latent_z, class_num, device).to(device)

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
        
        self.liner1 = nn.Linear(input_image + class_num, 512)
        self.liner2 = nn.Linear(512, 256)
        self.mu_liner =  nn.Linear(256, latent_z)
        self.logvar_liner =  nn.Linear(256, latent_z)

        self.selu = nn.SELU()                       #activation


    def forward(self, x, label):
        #先將原始image做flatten ex:[2,3,32,32]=>[2,3072]
        x = torch.flatten(x,start_dim=1)
        #print('image flatten shape',x.size())

        #cat 10類label標籤 [2,3072] + [2,10] => [2,3082]
        label_one_hot = self.one_hot(label, self.class_num).float().to(self.device)  # 使用 torch.eye() 进行 one-hot 编码
        #print(label_one_hot.size())
        
        x = torch.cat([x,label_one_hot],dim=1)  
        #print('cat image + label',x.size())
        x = self.liner1(x)
        x = self.selu(x)
        x = self.liner2(x)
        x = self.selu(x)

        z_mu = self.mu_liner(x)
        z_logvar = self.logvar_liner(x)
        
        return z_mu, z_logvar
    
    def one_hot(self, labels, num_classes):
        # label的shape是[batch,1(因為只有數字0~9,10類)] 要變成one hot 就要先創[0,0,0,0,0,0,0,0,0,0]
        one_hot = torch.zeros(labels.size(0), num_classes).to(self.device)
        #print(one_hot.size())
        #print(one_hot)
        # 使用 scatter_ 把label數值填上對應位置
        #ex: label=1  => [0,1,0,0,0,0,0,0,0,0] 
        #labels = labels.to(torch.int64)
        #print(one_hot.size())
        #print(labels.size())
        labels = torch.unsqueeze(labels,1).to(self.device)
        one_hot.scatter_(1, labels, 1)  #X.scatter_的維度 要和index=labels維度相同
        #print(one_hot)
        return one_hot
    
    

class cVAE_Decoder(nn.Module):
    def __init__(self, input_image, latent_z, class_num, device):
        super(cVAE_Decoder,self).__init__()
        self.input_image = input_image
        self.latent_z = latent_z
        self.class_num = class_num 
        self.device = device
        #decoder 也是 latent z + label
        self.liner1 = nn.Linear(latent_z+class_num, 256)
        self.liner2 = nn.Linear(256, 512)
        self.liner3 = nn.Linear(512, input_image)

        self.selu = nn.SELU()                       #activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, label):

        label_one_hot = self.one_hot(label, self.class_num).float().to(self.device)  # 使用 torch.eye() 进行 one-hot 编码
        #print(label_one_hot.size())
        z = torch.cat([z,label_one_hot],dim=1)  

        z = self.liner1(z)
        z = self.selu(z)
        z = self.liner2(z)
        z = self.selu(z)
        z = self.liner3(z)
        
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
        return one_hot

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    input_image = 3*32*32
    latent_z = 256
    class_num = 10
    model = cVAE_MLP(input_image = input_image,
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