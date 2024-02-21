import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
from torchsummary import summary
import math
#from torchinfo import summary

class U_net_Encoder(nn.Module):
    def __init__(self) -> None:
        super(U_net_Encoder,self).__init__()

    def forward(self,x):
        pass
        
class U_net_Decoder(nn.Module):
    def __init__(self) -> None:
        super(U_net_Decoder,self).__init__()

    def forward(self,x):
        pass

class SELayer(nn.Module):
    '''
        SE Block做chaeenl attention\n
        `mode`:有 'avg_pool'、'max_pool' 2種\n
        `channel`: 一開始進入MLP的通道數量\n
        `reduction`:MLP中間的過度filter size ratio, 預設16\n
        '''
    def __init__(self, mode, channel, reduction=16):
        super(SELayer, self).__init__()
        self.mode = mode
        #兩種初始化pooling方式
        #變成1x1xC 不同Channel做pooling
        if self.mode == 'avg_pool':
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        elif self.mode == 'max_pool':
            self.global_avg_pool = nn.AdaptiveMaxPool2d(1)
        #取得每個channel的權重
        self.fc = nn.Sequential(
            nn.Linear(in_features=channel,
                      out_features=channel//reduction,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel//reduction,
                      out_features=channel,
                      bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        B, C, H, W = x.size()
        #[B, C, H, W] => global_avg_pool:[B, C, 1, 1] => view:[B, C] 
        global_avg_pool = self.global_avg_pool(x).view(B,C)
        #經過mlp得到每個channel的權重後要變回[B, C, 1, 1]
        mlp_channel_attention = self.fc(global_avg_pool).view(B, C, 1, 1)
        #return 原來的input 乘上 channel_attention權重
        return x * mlp_channel_attention.expand_as(x) 

class ECALayer(nn.Module):
    '''
    SE Block參數精簡版 將MLP改成1維cnn 做chaeenl attention\n
        `mode`:有 'avg_pool'、'max_pool' 2種\n
        `channel`: 一開始進入MLP的通道數量\n
        `gamma`:改變channel數量和kernel大小的比例\n
        `b`:改變channel數量和kernal大小的比例\n
        k = |(log(C) + b)/gamma| 一定要奇數
    '''
    def __init__(self, channel, gamma=2, b=1):
        super(ECALayer,self).__init__()
        k = int((abs(math.log(channel, 2) + b))/gamma)
        k = k if k%2==1 else k+1   #一定要奇數
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=1,
                              kernel_size=k,
                              padding=(k-1),
                              )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        avg_pool = self.avg_pool(x) #現在維度是[B,C,1,1]
        #avg_pool.sequeeze(-1)=[B,C,1] => tranpose(-1,-2):[B,1,C]  => tranpose(-1,-2):[B,C,1] => unsequeeze(-1):[B,C,1,1]
        conv = self.conv(avg_pool.sequeeze(-1).tranpose(-1,-2)).transpose(-1,-2).unsequeeze(-1)
        sigmoid = self.sigmoid(conv)
        return x*sigmoid

class CBAMLayer(nn.Module):
    '''
    CBAM Block做chaeenl and spatial attention\n
    `kernel_size`:spatial attention的kernel_size, 預設7\n
    `channel`: 一開始進入MLP的通道數量\n
    `reduction`:MLP中間的過度filter size ratio, 預設16\n
    '''
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAMLayer, self).__init__()
        #channel attention
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=channel,
                      out_features=channel//reduction,
                      bias=False
                      ),
            nn.ReLU(),
            nn.Linear(in_features=channel//reduction,
                      out_features=channel,
                      bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        #spatial attention
        self.conv = nn.Conv2d(in_channels=2,
                              out_channels=1,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=kernel_size//2,
                              bias=False)
    def forward(self,x):
        #channel
        max_pool = self.max_pool(x)
        max_pool_flatten = max_pool.view(max_pool.size(0),-1)   #從B,C,H,W => B,C*H*W
        max_pool_mlp = self.mlp(max_pool_flatten)
        avg_pool = self.avg_pool(x)
        avg_pool_flatten = avg_pool.view(avg_pool.size(0),-1)   #從B,C,H,W => B,C*H*W
        avg_pool_mlp = self.mlp(avg_pool_flatten)               
        pooling_add_sig = self.sigmoid(max_pool_mlp+avg_pool_mlp)
        channel_weight = pooling_add_sig.view(x.size(0),x.size(1),1,1)
        channel_out = channel_weight * x
        #spatial 從channel attention輸出的channel_out做spatial attention
        max_pool_s,_ = torch.max(channel_out, dim=1, keepdim=True)
        avg_pool_s = torch.mean(channel_out, dim=1, keepdim=True)
        out = torch.cat((max_pool_s,avg_pool_s),dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        out = out * channel_out
        return out


class DoubleConv(nn.Module):
    """ |conv2d -> [BN] -> relu |* 2 \n
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 ) -> None:
        super(DoubleConv, self).__init__()
        #注意:U-net在剛開始要upsample的時候 要先將feature map 1024->512, 因為要和short cut做concate
        if not mid_channels:
            mid_channels = out_channels
        
        self.doubleConv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),#inplace=True能減少變數儲存空間
            nn.Conv2d(in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.doubleConv2d(x)

class DownSample(nn.Module):
    """
    DoubleConv->max pooling組成DownSample\n
    U-net down-conv:2x2
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ) -> None:
        super(DownSample,self).__init__()
        self.maxpooling_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=in_channels,
                       out_channels=out_channels)
        )
    def forward(self,x):
        return self.maxpooling_conv(x)

class UpSample(nn.Module):
    """
    UpSample->DoubleConv組成UpSample\n
    UpSample方式有2種nn.Upsample or nn.ConvTranspose2d\n
    nn.Upsample:不用訓練參數,直接用插值(速度較快)\n
    nn.ConvTranspose2d:逆捲積,需要訓練\n
    U-net up-conv:2x2\n
    `注意:`從DownSample過來的shortcut img shape可能會不同, 將這邊的UpSample的shape補成和shortcut shape一樣大小
    """
    def __init__(self,in_channels, out_channels, bilinear=True) -> None:
        super(UpSample,self).__init__()
        if bilinear:
            
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.doubleConv2d = DoubleConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           mid_channels=in_channels//2)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels=in_channels,
                                               out_channels=in_channels//2,
                                               kernel_size=2,
                                               stride=2)
            self.doubleConv2d = DoubleConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           )
        
    def forward(self,x_fromDown,x_up):
        #x_fromDown: short cut過來的
        x_up = self.upsample(x_up)#做upsample
        #size 依序是 Channel, Height, Width
        diffY = x_fromDown.size()[2] - x_up.size()[2]
        diffX = x_fromDown.size()[3] - x_up.size()[3]

        x_up = F.pad(x_up,[diffX//2, diffX - diffX//2,
                           diffY//2, diffY - diffY//2])
        x = torch.cat([x_fromDown, x_up], dim=1)#把skip connection和upsample concate後做double conv2d
        x = self.doubleConv2d(x)
        return x
   
class OutConv(nn.Module):
    '''
    U-net輸出 \n
    U-net out-conv:1x1 
    '''
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    '''
    model = SELayer(mode='avg_pool',
                    channel=64,
                    reduction=16).to(device)
    print(model)
    '''
    model = CBAMLayer(kernel_size=7,
                    channel=64,
                    reduction=16).to(device)
    #print(summary(model, input_size=(64, 224, 224)))

    fmap = torch.randn((4, 64, 224, 224)).to(device)
    print(model(fmap))

    print(model)
    print(summary(model, input_size=(64,512,512)))