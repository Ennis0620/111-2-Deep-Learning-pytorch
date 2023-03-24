import torch,torchvision
import torch.nn as nn
from dataset_loader import dataset_loader
import numpy as np
import cv2,os,tqdm,math
import matplotlib.pyplot as plt

class Grad_cam(nn.Module):
    def __init__(self,model):
        super(Grad_cam,self).__init__()
        self.model = model
        self.layers = list(model.children())
        self.grad_block = []
        self.fmap_block = []

    def backward_hook(self,
                      module,
                      grad_in,
                      grad_out):
        self.grad_block.append(grad_out[0].detach())
    
    def forward_hook(self,
                     module,
                     input,
                     output):
        self.fmap_block.append(output)
    
    def show_cam_on_img(self,
                        img,
                        mask,
                        save_path):
        '''
        將grad_cam的注意力mask畫在原本的img上
        '''
        #H,W,_ = img.shape
        #resize_h,resize_w = 128,128

        img = ((img + 1)/2)*255 #unnormalize to 0~255
        img = np.uint8(img.numpy())
        #print('img:',np.max(img),np.min(img))
        heatmap = cv2.applyColorMap(np.uint8(255*mask),cv2.COLORMAP_JET)
        #print('heatmap:',np.max( heatmap),np.min( heatmap))
        img = img.transpose(1,2,0)
        #print('img shape:',img.shape)
        cam_img= np.uint8(0.3*heatmap + 0.7*img)
        #print('cam_img range:',np.min(cam_img),'~',np.max(cam_img))
        img = cv2.resize(img,(64,64))
        heatmap = cv2.resize(heatmap,(64,64))
        cam_img = cv2.resize(cam_img,(64,64))
    
        space = np.ones((cam_img.shape[0],5,cam_img.shape[2]))
        con = np.concatenate([img,space,heatmap,space,cam_img],axis=1)
        #cv2.imwrite(save_path,con)
        
        #用plt show圖片
        plt.figure()
        con = cv2.cvtColor(np.float32(con/255),cv2.COLOR_BGR2RGB)#BGR->RGB pixel range:0~1
        im_ratio = con.shape[0]/con.shape[1]
        plt.imshow(con,cmap='jet')
        plt.colorbar(fraction=0.047*im_ratio)
        plt.savefig(save_path,bbox_inches='tight')

    def img_preprocess(img):
        '''
        '''
        img_in = img.copy()    
        img_in = img[:,:,::-1]
        img_in = np.ascontiguousarray(img_in)
        transform = torchvision.transforms.Compose(
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        )
        img_out = transform(img_in)
        img_out = img_out.squeeze(0)
        return img_out

    def imshow(self,
               img,
               save_path):
        '''
        顯示test_loader的影像
        '''
        img = img / 2 + 0.5 #unnormalizate
        npimg = img.numpy()
        npimg = np.transpose(npimg,(1,2,0))
        plt.imshow(npimg)
        plt.savefig(f'{save_path}')
        #plt.show()

    def comp_class_vec(self,output_vec, index=None):
        """
        计算类向量
        :param ouput_vec: tensor
        :param index: int，指定类别
        :return: tensor
        """
        if not index:
            index = np.argmax(output_vec.cpu().data.numpy())
        else:
            index = np.array(index)
        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
        one_hot.requires_grad = True
        class_vec = torch.sum(one_hot.to(device) * output_vec)  # one_hot = 11.8605
        return class_vec        

    def generate_grad_cam_focus(self,
                                fmap,
                                grad):
        '''
        依gradient和featuremap生成cam \n
        `fmap`:np array [C,H,W] \n
        `grad`:np array [C,H,W] \n
        @return \n
        np.array [H,W]
        '''
        
        cam = np.zeros(fmap.shape[1:],
                        dtype = np.float32)
        #print('cam zero:',cam.shape)
        weights = np.mean(grad,axis=(1,2))
        #print('weights:',weights.shape)

        for i,w in enumerate(weights):
            #print('w:',w)
            cam += w * fmap[i, :, :]
 
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (32,32))
        cam -= np.min(cam)
        cam /= np.max(cam)
        
        return cam
    
        
if __name__ == "__main__":
    

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    CURRENT_PATH = os.path.dirname(__file__)
    #BEST_WEIGHT_NAME = sorted(os.listdir(f'{CURRENT_PATH}/model_weight'))[0]
    BEST_WEIGHT_NAME = f'epoch_91_trainLoss_0.5959_trainAcc_78.99_valLoss_0.5785_valAcc_80.05.pth'
    SAVE_MODELS_PATH = f'{CURRENT_PATH}/model_weight/5conv_2bn/{BEST_WEIGHT_NAME}'
    SAVE_GRAD_CAM_VISUAL = f'{CURRENT_PATH}/grad_cam_visual'

    DATASET_NAME = 'CIFAR10'
    SPLIT_RATIO = 0.2
    SHUFFLE_DATASET = False
    BATCH_SIZE=49

    LOSS = nn.CrossEntropyLoss()

    DL = dataset_loader(dataset_name=DATASET_NAME,
                        valid_split_ratio=SPLIT_RATIO,
                        shuffle=SHUFFLE_DATASET,
                        batch_size=BATCH_SIZE)
    

    test_loader = DL.test_dataloader(show_samples=False)
    classes = DL.label_names
    
    net = torch.load(SAVE_MODELS_PATH)
    print(net)

    test_iter = iter(test_loader)
    img,label = next(test_iter)
    show_img = torchvision.utils.make_grid(img,
                                           nrow=int(math.sqrt(BATCH_SIZE)))
    #print('show_img shape:',show_img.shape)
    #img = img.to(device)
    #label = label.to(device)
    
    
    for idx,(per_img,per_label) in tqdm.tqdm(enumerate(zip(img,label))):
        
        grad_cam = Grad_cam(model=net)    
        
        #grad-cam需要最後一捲積層前向&反向傳播
        
        grad_cam.model.conv5.register_forward_hook(grad_cam.forward_hook)
        grad_cam.model.conv5.register_backward_hook(grad_cam.backward_hook)
        #print(grad_cam.model.conv2)
    
        grad_cam.imshow(show_img,
                    save_path=f'{SAVE_GRAD_CAM_VISUAL}/origin.jpg')
    
        feed2network_img = np.expand_dims(per_img,axis=0)#給network必須為4維(張數,C,H,W)
        feed2network_img = torch.from_numpy(feed2network_img).to(device)
        feed2network_label = np.expand_dims(per_label,axis=0)#給network必須為4維(張數,C,H,W)
        feed2network_label = torch.from_numpy(feed2network_label).to(device)
        #forward
        net.eval()
        out = net(feed2network_img)
        #print("predict: {}".format(classes[idx]))
        #backward
        net.zero_grad()
        #loss = LOSS(out,feed2network_label)
        loss = grad_cam.comp_class_vec(out)
        loss.backward()

        test_pred = out.argmax(dim=1)                 #取得預測的最大值

        grad = grad_cam.grad_block[0].cpu().data.numpy().squeeze()
        fmap = grad_cam.fmap_block[0].cpu().data.numpy().squeeze()
        cam = grad_cam.generate_grad_cam_focus(fmap=fmap,
                                            grad=grad)
        #print(fmap.shape)
        #print(grad.shape)
        #print(cam.shape)
        predict = classes[test_pred.cpu().numpy()[0]]
        real_ans = classes[per_label]
        #print(predict)

        save_path = f'{SAVE_GRAD_CAM_VISUAL}/pred_{predict}__real_{real_ans}_{idx}.jpg'
        grad_cam.show_cam_on_img(img=per_img,
                                mask=cam,
                                save_path=save_path)


    
