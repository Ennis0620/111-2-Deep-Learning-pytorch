import torch.nn.functional as F
import torch.nn as nn
import torch
from torchsummary import summary

class network(nn.Module):
    def __init__(self,num_classes: int):
        super(network, self).__init__()
        self.num_classes = num_classes
        # in[N, 3, 32, 32] => out[N, 32, 30, 30]
        '''
        self.conv1 = nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=3,
                                padding='same')
        self.conv2 = nn.Conv2d(in_channels=64,
                                out_channels=32,
                                kernel_size=3,
                                padding='same')
        self.conv3 = nn.Conv2d(in_channels=32,
                                out_channels=32,
                                kernel_size=3,
                                padding='same')
        self.conv4 = nn.Conv2d(in_channels=32,
                                out_channels=16,
                                kernel_size=3,
                                padding='same')
        self.conv5 = nn.Conv2d(in_channels=16,
                                out_channels=16,
                                kernel_size=3,
                                padding='valid')
        '''
        self.conv1 = nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=3,
                                padding='same')
        self.conv2 = nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                padding='same')
        self.conv3 = nn.Conv2d(in_channels=128,
                                out_channels=128,
                                kernel_size=3,
                                padding='same')
        self.conv4 = nn.Conv2d(in_channels=128,
                                out_channels=256,
                                kernel_size=3,
                                padding='same')
        self.conv5 = nn.Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=3,
                                padding='valid')

        self.conv1_k_5 = nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=5,
                                padding='same')
        self.conv2_k_5 = nn.Conv2d(in_channels=128,
                                out_channels=128,
                                kernel_size=5,
                                padding='same')
        self.conv3_k_5 = nn.Conv2d(in_channels=128,
                                out_channels=256,
                                kernel_size=5,
                                padding='same')

        self.bottle_neck = nn.Conv2d(in_channels=128,
                                out_channels=32,
                                kernel_size=1,
                                padding='valid')
        self.bottle_neck2 = nn.Conv2d(in_channels=32,
                                out_channels=32,
                                kernel_size=1,
                                padding='valid')
        

        self.dropout = nn.Dropout(p=0.25)
        self.pool = nn.MaxPool2d(kernel_size=2,
                                  stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2,
                                  stride=1,
                                  )
        self.pool3 = nn.MaxPool2d(kernel_size=3,
                                  stride=3,
                                  )
        self.bn16 = nn.BatchNorm2d(num_features=16)
        self.bn32 = nn.BatchNorm2d(num_features=32)
        self.bn64 = nn.BatchNorm2d(num_features=64)
        self.bn128= nn.BatchNorm2d(num_features=128)
        self.bn256 = nn.BatchNorm2d(num_features=256)
        self.bn512 = nn.BatchNorm2d(num_features=512)


        self.bn = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(in_features=512*7*7,
                             out_features=self.num_classes)

        #3 conv
        #self.fc1 = nn.Linear(in_features=16*2*2 ,
        #                     out_features=128)
        self.fc2 = nn.Linear(in_features=128 ,
                             out_features=32)
        self.fc3 = nn.Linear(in_features=32 ,
                             out_features=self.num_classes)
    
    def forward(self, layer):

        layer = self.pool2(F.relu(self.conv1(layer)))
        layer = self.bn64(layer)
        layer = self.dropout(layer)
        temp128_31_31 = layer

        layer_g1 = self.pool2(F.relu(self.conv1_k_5(temp128_31_31)))
        layer_g1 = self.bn128(layer_g1)
        layer_g1 = self.dropout(layer_g1)
        layer_g1 = self.pool(F.relu(self.conv2_k_5(layer_g1)))
        layer_g1 = self.bn128(layer_g1)
        layer_g1 = self.dropout(layer_g1)
        layer_g1 = self.pool(F.relu(self.conv3_k_5(layer_g1)))
        layer_g1 = self.dropout(layer_g1)

        layer_g2 = self.pool2(F.relu(self.conv2(temp128_31_31)))
        layer_g2 = self.bn128(layer_g2)
        layer_g2 = self.dropout(layer_g2)
        layer_g2 = self.pool(F.relu(self.conv3(layer_g2)))
        layer_g2 = self.bn128(layer_g2)
        layer_g2 = self.dropout(layer_g2)
        layer_g2 = self.pool(F.relu(self.conv4(layer_g2)))
        layer_g2 = self.dropout(layer_g2)

        layer = torch.cat((layer_g1,layer_g2),dim=1)
        layer = self.bn512(layer)
        
        #temp = self.pool(self.bottle_neck(temp128_31_31))
        #temp = self.pool(self.bottle_neck2(temp))
        
        #layer = layer + temp
        layer = self.dropout(layer)
        layer = torch.flatten(layer, 1)
        layer = self.fc1(layer)
        return layer

        #layer = self.bn64(layer)
        #layer = self.dropout(layer)
        #layer = F.relu(self.conv3(layer))
        #layer = self.bn2(layer)
        #layer = F.relu(self.conv4(layer))
        #layer = self.bn3(layer)
        #layer = F.relu(self.conv5(layer))
        
        #layer = F.relu(self.fc1(layer))
        #layer = F.relu(self.fc2(layer))
        # = self.fc3(layer)


class ResBlock(nn.Module):
    def __init__(self,
                in_channel,
                out_channel,
                stride=1,
                down_sample='None',
                ):
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.down_sample = down_sample

    def forward(self,layer):
        residual = layer     #開始進入resblock前的參數值
        out = self.conv1(layer)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample:
            residual = self.down_sample(layer)
        out += residual       #結束resblock要加上剛開始進來的 skip connection
        out = self.relu(out)
        return out

class CNN(nn.Module):
  def __init__(self, num_classes: int):
    super(CNN, self).__init__()
    self.num_classes = num_classes

    # in[N, 3, 32, 32] => out[N, 16, 16, 16]
    self.conv1 = nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
        ),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2)
    )
    # in[N, 16, 16, 16] => out[N, 32, 8, 8]
    self.conv2 = nn.Sequential(
        nn.Conv2d(16, 32, 5, 1, 2),
        nn.ReLU(True),
        nn.MaxPool2d(2)
    )
    # in[N, 32 * 8 * 8] => out[N, 128]
    self.fc1 = nn.Sequential(
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(True)
    )
    # in[N, 128] => out[N, 64]
    self.fc2 = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(True)
    )
    # in[N, 64] => out[N, 10]
    self.out = nn.Linear(64, self.num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1) # [N, 32 * 8 * 8]
    x = self.fc1(x)
    x = self.fc2(x)
    output = self.out(x)
    return output

if __name__ == '__main__':
    #GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:',device)
    net = network(10).to(device=device)
    print(net)
    print(summary(net, input_size=(3,32,32)))