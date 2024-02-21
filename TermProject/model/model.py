import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
from torchsummary import summary
from .model_block import *

class U_net(nn.Module):
    """
    `U-net`\n
    U-net架構 4個down+4個up\n
    encoder : feature map = [64,128,256,512,1024]\n
    decoder : feature map = [64,128,256,512,1024]\n
    `n_channels`:影像通道數
    `n_classes`:分幾類
    `bilinear`:用nn.Sample:True, 用nn.ConvTranspose2d:False
    """
    def __init__(self,n_channels, n_classes, bilinear=False) -> None:
        super(U_net,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #U-net架構 4個down+4個up
        self.init = DoubleConv(in_channels=n_channels,
                               out_channels=64)
        self.down1 = DownSample(in_channels=64,
                                out_channels=128)
        self.down2 = DownSample(in_channels=128,
                                out_channels=256)
        self.down3 = DownSample(in_channels=256,
                                out_channels=512)
        #如果要用nn.Sample, 設定factor = 2
        #如果使用nn.ConvTranspose2d, 設定factor = 1
        factor = 2 if self.bilinear else 1 
        #down4後會接Upsample所以要先
        self.down4 = DownSample(in_channels=512,
                                out_channels=1024//factor)
        self.up1 = UpSample(in_channels=1024, 
                            out_channels= 512//factor,
                             bilinear=bilinear)
        self.up2 = UpSample(in_channels=512, 
                            out_channels= 256//factor,
                             bilinear=bilinear)
        self.up3 = UpSample(in_channels=256, 
                            out_channels= 128//factor,
                             bilinear=bilinear)
        self.up4 = UpSample(in_channels=128, 
                            out_channels= 64,
                             bilinear=bilinear)
        self.out = OutConv(in_channels= 64,
                           out_channels= n_classes)

    def forward(self,x):
        x1 = self.init(x)  #feature map = 64
        x2 = self.down1(x1)#feature map = 128
        x3 = self.down2(x2)#feature map = 256
        x4 = self.down3(x3)#feature map = 512
        x5 = self.down4(x4)#feature map = 1024(使用nn.ConvTranspose2d), feature map = 512(使用nn.Sample)
        x = self.up1(x4,x5)#feature map = 512(使用nn.ConvTranspose2d), feature map = 256(使用nn.Sample)
        x = self.up2(x3,x) #feature map = 256(使用nn.ConvTranspose2d), feature map = 128(使用nn.Sample)
        x = self.up3(x2,x) #feature map = 128(使用nn.ConvTranspose2d), feature map = 64(使用nn.Sample)
        x = self.up4(x1,x) #feature map = 64(使用nn.ConvTranspose2d), feature map = 64(使用nn.Sample)
        out = self.out(x)  #輸出classes張mask
        return out

class SE_U_net(nn.Module):
    """
    `SE Attention U-net`\n
    U-net架構 4個down+4個up\n
    encoder : feature map = [64,128,256,512,1024]\n
    decoder : feature map = [64,128,256,512,1024]\n
    `n_channels`:影像通道數\n
    `n_classes`:分幾類\n
    `mode`:SE Block的pool方式, `mode`:有 'avg_pool'、'max_pool' 2種\n
    `bilinear`:用nn.Sample:True, 用nn.ConvTranspose2d:False
    """
    def __init__(self,n_channels, n_classes, mode, bilinear=False) -> None:
        super(SE_U_net,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.mode = mode
        #U-net架構 4個down+4個up
        self.init = DoubleConv(in_channels=n_channels,
                               out_channels=64)
        self.down1 = DownSample(in_channels=64,
                                out_channels=128)
        self.SE_down1 = SELayer(mode=self.mode,
                                channel=128)
        
        self.down2 = DownSample(in_channels=128,
                                out_channels=256)
        self.SE_down2 = SELayer(mode=self.mode,
                                channel=256)
        self.down3 = DownSample(in_channels=256,
                                out_channels=512)
        self.SE_down3 = SELayer(mode=self.mode,
                                channel=512)
        #如果要用nn.Sample, 設定factor = 2
        #如果使用nn.ConvTranspose2d, 設定factor = 1
        factor = 2 if self.bilinear else 1 
        #down4後會接Upsample所以要先
        self.down4 = DownSample(in_channels=512,
                                out_channels=1024//factor)
        self.SE_down4 = SELayer(mode=self.mode,
                                channel=1024//factor)
        
        self.up1 = UpSample(in_channels=1024, 
                            out_channels= 512//factor,
                             bilinear=bilinear)
        self.SE_up1 = SELayer(mode=self.mode,
                                channel=512//factor)
        
        self.up2 = UpSample(in_channels=512, 
                            out_channels= 256//factor,
                             bilinear=bilinear)
        self.SE_up2 = SELayer(mode=self.mode,
                                channel=256//factor)
        
        self.up3 = UpSample(in_channels=256, 
                            out_channels= 128//factor,
                             bilinear=bilinear)
        self.SE_up3 = SELayer(mode=self.mode,
                                channel=128//factor)
        
        self.up4 = UpSample(in_channels=128, 
                            out_channels= 64,
                             bilinear=bilinear)
        self.SE_up4 = SELayer(mode=self.mode,
                                channel=64)

        self.out = OutConv(in_channels= 64,
                           out_channels= n_classes)
        
    def forward(self,x):
        x1 = self.init(x)  #feature map = 64

        x2 = self.down1(x1)#feature map = 128
        x2 = self.SE_down1(x2)

        x3 = self.down2(x2)#feature map = 256
        x3 = self.SE_down2(x3)

        x4 = self.down3(x3)#feature map = 512
        x4 = self.SE_down3(x4)

        x5 = self.down4(x4)#feature map = 1024(使用nn.ConvTranspose2d), feature map = 512(使用nn.Sample)
        x5 = self.SE_down4(x5)
        #------------------------------------------------
        x = self.up1(x4,x5)#feature map = 512(使用nn.ConvTranspose2d), feature map = 256(使用nn.Sample)
        x = self.SE_up1(x)

        x = self.up2(x3,x) #feature map = 256(使用nn.ConvTranspose2d), feature map = 128(使用nn.Sample)
        x = self.SE_up2(x)

        x = self.up3(x2,x) #feature map = 128(使用nn.ConvTranspose2d), feature map = 64(使用nn.Sample)
        x = self.SE_up3(x)

        x = self.up4(x1,x) #feature map = 64(使用nn.ConvTranspose2d), feature map = 64(使用nn.Sample)
        x = self.SE_up4(x)

        out = self.out(x)  #輸出classes張mask
        return out

class SE_U_net_only_scout(nn.Module):
    """
    `SE Attention U-net`\n
    U-net架構 4個down+4個up\n
    encoder : feature map = [64,128,256,512,1024]\n
    decoder : feature map = [64,128,256,512,1024]\n
    `n_channels`:影像通道數\n
    `n_classes`:分幾類\n
    `mode`:SE Block的pool方式, `mode`:有 'avg_pool'、'max_pool' 2種\n
    `bilinear`:用nn.Sample:True, 用nn.ConvTranspose2d:False
    """
    def __init__(self,n_channels, n_classes, mode, bilinear=False) -> None:
        super(SE_U_net_only_scout,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.mode = mode
        #U-net架構 4個down+4個up
        self.init = DoubleConv(in_channels=n_channels,
                               out_channels=64)
        self.down1 = DownSample(in_channels=64,
                                out_channels=128)
        self.SE_down1 = SELayer(mode=self.mode,
                                channel=128)
        
        self.down2 = DownSample(in_channels=128,
                                out_channels=256)
        self.SE_down2 = SELayer(mode=self.mode,
                                channel=256)
        self.down3 = DownSample(in_channels=256,
                                out_channels=512)
        self.SE_down3 = SELayer(mode=self.mode,
                                channel=512)
        #如果要用nn.Sample, 設定factor = 2
        #如果使用nn.ConvTranspose2d, 設定factor = 1
        factor = 2 if self.bilinear else 1 
        #down4後會接Upsample所以要先
        self.down4 = DownSample(in_channels=512,
                                out_channels=1024//factor)
        self.SE_down4 = SELayer(mode=self.mode,
                                channel=1024//factor)
        
        self.up1 = UpSample(in_channels=1024, 
                            out_channels= 512//factor,
                             bilinear=bilinear)
        self.SE_up1 = SELayer(mode=self.mode,
                                channel=512//factor)
        
        self.up2 = UpSample(in_channels=512, 
                            out_channels= 256//factor,
                             bilinear=bilinear)
        self.SE_up2 = SELayer(mode=self.mode,
                                channel=256//factor)
        
        self.up3 = UpSample(in_channels=256, 
                            out_channels= 128//factor,
                             bilinear=bilinear)
        self.SE_up3 = SELayer(mode=self.mode,
                                channel=128//factor)
        
        self.up4 = UpSample(in_channels=128, 
                            out_channels= 64,
                             bilinear=bilinear)
        self.SE_up4 = SELayer(mode=self.mode,
                                channel=64)

        self.out = OutConv(in_channels= 64,
                           out_channels= n_classes)
        
    def forward(self,x):
        x1 = self.init(x)  #feature map = 64

        x2 = self.down1(x1)#feature map = 128
        x2_att = self.SE_down1(x2)

        x3 = self.down2(x2)#feature map = 256
        x3_att = self.SE_down2(x3)

        x4 = self.down3(x3)#feature map = 512
        x4_att = self.SE_down3(x4)

        x5 = self.down4(x4)#feature map = 1024(使用nn.ConvTranspose2d), feature map = 512(使用nn.Sample)
        x5_att = self.SE_down4(x5)
        #------------------------------------------------
        x = self.up1(x4_att,x5_att)#feature map = 512(使用nn.ConvTranspose2d), feature map = 256(使用nn.Sample)
        

        x = self.up2(x3_att,x) #feature map = 256(使用nn.ConvTranspose2d), feature map = 128(使用nn.Sample)
        

        x = self.up3(x2_att,x) #feature map = 128(使用nn.ConvTranspose2d), feature map = 64(使用nn.Sample)
        

        x = self.up4(x1,x) #feature map = 64(使用nn.ConvTranspose2d), feature map = 64(使用nn.Sample)
        

        out = self.out(x)  #輸出classes張mask
        return out

class CBAM_U_net(nn.Module):
    """
    `CBA Attention U-net`\n
    U-net架構 4個down+4個up\n
    encoder : feature map = [64,128,256,512,1024]\n
    decoder : feature map = [64,128,256,512,1024]\n
    `n_channels`:影像通道數\n
    `n_classes`:分幾類\n
    `bilinear`:用nn.Sample:True, 用nn.ConvTranspose2d:False
    """
    def __init__(self,n_channels, n_classes, bilinear=False) -> None:
        super(CBAM_U_net,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        #U-net架構 4個down+4個up
        self.init = DoubleConv(in_channels=n_channels,
                               out_channels=64)
        self.down1 = DownSample(in_channels=64,
                                out_channels=128)
        self.CBAM_down1 = CBAMLayer(channel=128)
        
        self.down2 = DownSample(in_channels=128,
                                out_channels=256)
        self.CBAM_down2 = CBAMLayer(channel=256)
        self.down3 = DownSample(in_channels=256,
                                out_channels=512)
        self.CBAM_down3 = CBAMLayer(channel=512)
        #如果要用nn.Sample, 設定factor = 2
        #如果使用nn.ConvTranspose2d, 設定factor = 1
        factor = 2 if self.bilinear else 1 
        #down4後會接Upsample所以要先
        self.down4 = DownSample(in_channels=512,
                                out_channels=1024//factor)
        self.CBAM_down4 = CBAMLayer(channel=1024//factor)
        self.up1 = UpSample(in_channels=1024, 
                            out_channels= 512//factor,
                             bilinear=bilinear)
        self.CBAM_up1 = CBAMLayer(channel=512//factor)

        self.up2 = UpSample(in_channels=512, 
                            out_channels= 256//factor,
                             bilinear=bilinear)
        self.CBAM_up2 = CBAMLayer(channel=256//factor)
        
        self.up3 = UpSample(in_channels=256, 
                            out_channels= 128//factor,
                             bilinear=bilinear)
        self.CBAM_up3 = CBAMLayer(channel=128//factor)
        
        self.up4 = UpSample(in_channels=128, 
                            out_channels= 64,
                             bilinear=bilinear)
        self.CBAM_up4 = CBAMLayer(channel=64)

        self.out = OutConv(in_channels= 64,
                           out_channels= n_classes)
        
    def forward(self,x):
        x1 = self.init(x)  #feature map = 64

        x2 = self.down1(x1)#feature map = 128
        x2 = self.CBAM_down1(x2)

        x3 = self.down2(x2)#feature map = 256
        x3 = self.CBAM_down2(x3)

        x4 = self.down3(x3)#feature map = 512
        x4 = self.CBAM_down3(x4)

        x5 = self.down4(x4)#feature map = 1024(使用nn.ConvTranspose2d), feature map = 512(使用nn.Sample)
        x5 = self.CBAM_down4(x5)
        #------------------------------------------------
        x = self.up1(x4,x5)#feature map = 512(使用nn.ConvTranspose2d), feature map = 256(使用nn.Sample)
        x = self.CBAM_up1(x)

        x = self.up2(x3,x) #feature map = 256(使用nn.ConvTranspose2d), feature map = 128(使用nn.Sample)
        x = self.CBAM_up2(x)

        x = self.up3(x2,x) #feature map = 128(使用nn.ConvTranspose2d), feature map = 64(使用nn.Sample)
        x = self.CBAM_up3(x)

        x = self.up4(x1,x) #feature map = 64(使用nn.ConvTranspose2d), feature map = 64(使用nn.Sample)
        x = self.CBAM_up4(x)

        out = self.out(x)  #輸出classes張mask
        return out

class CBAM_U_net_only_scout(nn.Module):
    """
    `CBA Attention U-net`\n
    U-net架構 4個down+4個up\n
    encoder : feature map = [64,128,256,512,1024]\n
    decoder : feature map = [64,128,256,512,1024]\n
    `n_channels`:影像通道數\n
    `n_classes`:分幾類\n
    `bilinear`:用nn.Sample:True, 用nn.ConvTranspose2d:False
    """
    def __init__(self,n_channels, n_classes, bilinear=False) -> None:
        super(CBAM_U_net_only_scout,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        #U-net架構 4個down+4個up
        self.init = DoubleConv(in_channels=n_channels,
                               out_channels=64)
        self.down1 = DownSample(in_channels=64,
                                out_channels=128)
        self.CBAM_down1 = CBAMLayer(channel=128)
        
        self.down2 = DownSample(in_channels=128,
                                out_channels=256)
        self.CBAM_down2 = CBAMLayer(channel=256)
        self.down3 = DownSample(in_channels=256,
                                out_channels=512)
        self.CBAM_down3 = CBAMLayer(channel=512)
        #如果要用nn.Sample, 設定factor = 2
        #如果使用nn.ConvTranspose2d, 設定factor = 1
        factor = 2 if self.bilinear else 1 
        #down4後會接Upsample所以要先
        self.down4 = DownSample(in_channels=512,
                                out_channels=1024//factor)
        self.CBAM_down4 = CBAMLayer(channel=1024//factor)
        self.up1 = UpSample(in_channels=1024, 
                            out_channels= 512//factor,
                             bilinear=bilinear)
        self.CBAM_up1 = CBAMLayer(channel=512//factor)

        self.up2 = UpSample(in_channels=512, 
                            out_channels= 256//factor,
                             bilinear=bilinear)
        self.CBAM_up2 = CBAMLayer(channel=256//factor)
        
        self.up3 = UpSample(in_channels=256, 
                            out_channels= 128//factor,
                             bilinear=bilinear)
        self.CBAM_up3 = CBAMLayer(channel=128//factor)
        
        self.up4 = UpSample(in_channels=128, 
                            out_channels= 64,
                             bilinear=bilinear)
        self.CBAM_up4 = CBAMLayer(channel=64)

        self.out = OutConv(in_channels= 64,
                           out_channels= n_classes)
        
    def forward(self,x):
        x1 = self.init(x)  #feature map = 64

        x2 = self.down1(x1)#feature map = 128
        x2_att = self.CBAM_down1(x2)

        x3 = self.down2(x2)#feature map = 256
        x3_att = self.CBAM_down2(x3)

        x4 = self.down3(x3)#feature map = 512
        x4_att = self.CBAM_down3(x4)

        x5 = self.down4(x4)#feature map = 1024(使用nn.ConvTranspose2d), feature map = 512(使用nn.Sample)
        x5_att = self.CBAM_down4(x5)
        #------------------------------------------------
        x = self.up1(x4_att,x5_att)#feature map = 512(使用nn.ConvTranspose2d), feature map = 256(使用nn.Sample)
        
        x = self.up2(x3_att,x) #feature map = 256(使用nn.ConvTranspose2d), feature map = 128(使用nn.Sample)

        x = self.up3(x2_att,x) #feature map = 128(使用nn.ConvTranspose2d), feature map = 64(使用nn.Sample)
        
        x = self.up4(x1,x) #feature map = 64(使用nn.ConvTranspose2d), feature map = 64(使用nn.Sample)
        

        out = self.out(x)  #輸出classes張mask
        return out

class SE_U_net_Encoder_CBAM_U_net_scout(nn.Module):
    """
    `SE_U_net_Encoder_CBAM_U_net_scout`\n
    U-net架構 4個down+4個up\n
    encoder : feature map = [64,128,256,512,1024]\n
    decoder : feature map = [64,128,256,512,1024]\n
    `n_channels`:影像通道數\n
    `n_classes`:分幾類\n
    `bilinear`:用nn.Sample:True, 用nn.ConvTranspose2d:False
    """
    def __init__(self,n_channels, mode,n_classes, bilinear=False) -> None:
        super(SE_U_net_Encoder_CBAM_U_net_scout,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.mode = mode
        #U-net架構 4個down+4個up
        self.init = DoubleConv(in_channels=n_channels,
                               out_channels=64)
        self.down1 = DownSample(in_channels=64,
                                out_channels=128)
        self.SE_down1 = SELayer(mode=self.mode,
                                channel=128)
        self.CBAM_down1 = CBAMLayer(channel=128)
        
        self.down2 = DownSample(in_channels=128,
                                out_channels=256)
        self.CBAM_down2 = CBAMLayer(channel=256)
        self.SE_down2 = SELayer(mode=self.mode,
                                channel=256)
        self.down3 = DownSample(in_channels=256,
                                out_channels=512)
        self.CBAM_down3 = CBAMLayer(channel=512)
        self.SE_down3 = SELayer(mode=self.mode,
                                channel=512)
        #如果要用nn.Sample, 設定factor = 2
        #如果使用nn.ConvTranspose2d, 設定factor = 1
        factor = 2 if self.bilinear else 1 
        #down4後會接Upsample所以要先
        self.down4 = DownSample(in_channels=512,
                                out_channels=1024//factor)
        self.CBAM_down4 = CBAMLayer(channel=1024//factor)
        self.SE_up4 = SELayer(mode=self.mode,
                                channel=64)
        self.up1 = UpSample(in_channels=1024, 
                            out_channels= 512//factor,
                             bilinear=bilinear)
        self.CBAM_up1 = CBAMLayer(channel=512//factor)

        self.up2 = UpSample(in_channels=512, 
                            out_channels= 256//factor,
                             bilinear=bilinear)
        self.CBAM_up2 = CBAMLayer(channel=256//factor)
        
        self.up3 = UpSample(in_channels=256, 
                            out_channels= 128//factor,
                             bilinear=bilinear)
        self.CBAM_up3 = CBAMLayer(channel=128//factor)
        
        self.up4 = UpSample(in_channels=128, 
                            out_channels= 64,
                             bilinear=bilinear)
        self.CBAM_up4 = CBAMLayer(channel=64)

        self.out = OutConv(in_channels= 64,
                           out_channels= n_classes)
        
    def forward(self,x):
        x1 = self.init(x)  #feature map = 64

        x2 = self.down1(x1)#feature map = 128
        x2_att = self.CBAM_down1(x2)
        x2 = self.SE_down1(x2)

        x3 = self.down2(x2)#feature map = 256
        x3_att = self.CBAM_down2(x3)
        x3 = self.SE_down2(x3)

        x4 = self.down3(x3)#feature map = 512
        x4_att = self.CBAM_down3(x4)
        x4 = self.SE_down3(x4)

        x5 = self.down4(x4)#feature map = 1024(使用nn.ConvTranspose2d), feature map = 512(使用nn.Sample)
        x5_att = self.CBAM_down4(x5)
        #------------------------------------------------
        x = self.up1(x4_att,x5_att)#feature map = 512(使用nn.ConvTranspose2d), feature map = 256(使用nn.Sample)
        
        x = self.up2(x3_att,x) #feature map = 256(使用nn.ConvTranspose2d), feature map = 128(使用nn.Sample)

        x = self.up3(x2_att,x) #feature map = 128(使用nn.ConvTranspose2d), feature map = 64(使用nn.Sample)
        
        x = self.up4(x1,x) #feature map = 64(使用nn.ConvTranspose2d), feature map = 64(使用nn.Sample)
        

        out = self.out(x)  #輸出classes張mask
        return out



if __name__ == '__main__':
   #GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:',device)
    '''
    model = SE_U_net(
                  n_channels=3,
                  n_classes=3,
                  mode='avg_pool',
                  bilinear=1).to(device)

    '''
    model = U_net(n_channels=1,
                  n_classes=3,
                  bilinear=1).to(device)
    # model = SE_U_net(
    #               n_channels=1,
    #               n_classes=3,
    #               mode='avg_pool',
    #               bilinear=1).to(device)
    
    # model = CBAM_U_net(
    #               n_channels=1,
    #               n_classes=3,
    #               bilinear=1).to(device)
    
    print(model)
    print(summary(model, input_size=(1,224,224)))
    
