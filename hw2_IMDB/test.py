import os,shutil,re
import pandas as pd
import emoji,tqdm
import torch
import statistics
import numpy as np
import torch.nn as nn
from model import RNN_IMDB,LSTM_IMDB,RNN2_IMDB,LSTM2_IMDB,GRU_IMDB,GRU2_IMDB,RNN_IMDB_BCE
import matplotlib.pyplot as plt
from dataset_test import dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,ConfusionMatrixDisplay,classification_report


def w2v_dict(GLOVE_PATH):
        '''
        把GLOVE的詞和向量讀入成字典
        '''
        w2v_dic={}
        w2v_mean=[0 for i in range(100)]

        with open(GLOVE_PATH,'r',encoding="utf-8") as fp:
            all = fp.readlines()
            for idx,word in tqdm.tqdm(enumerate(all)):
                word_split = word.split(' ')
                word_key = word_split[0]
                word_vec = np.array(word_split[1:]).astype(float).tolist()
                w2v_dic.setdefault(word_key,word_vec)

                for vec_idx,vec in enumerate(word_split[1:]):
                    w2v_mean[vec_idx] += float(vec)
        #如果沒有出現在GLOVE中,把GLOVE中所有的取平均填充
        w2v_mean = np.array(w2v_mean)
        w2v_mean /= len(w2v_dic)
        #print('w2v_mean:',w2v_mean)
        return w2v_dic,w2v_mean.tolist()

def replace_w2v(w2v_dicts:dict,
                w2v_dicts_mean:list,
                seg_len:int,
                string:str,
                ):
    '''
    把string轉成vector
    '''
    return_list = []
    string = str(string)
    string_split = string.split()

    for word in string_split:
        #print('per word:',word)
        if word in w2v_dicts:
            return_list.append(w2v_dicts[word])
        else:
            return_list.append(w2v_dicts_mean)
            
        #確保在test階段load資料時seq長度不超過訓練時的長度
        if len(return_list)>= seg_len:
            return return_list
    #比訓練時的seq長度短 padding 成一樣維度
    if len(string_split)!=seg_len:
        for i in range(seg_len-len(string_split)):
            return_list.append(w2v_dicts_mean)

    return return_list

def seg_sentence(string):
    #斷詞
    pattern = r'[^\w\s]+'
    return re.sub(pattern, ' ', string)

def rm_stop_word(stop_word_set,string):
    '''
    去除停用字詞
    '''
    return_string = ""
    split_string = string.split()
    for idx,word in enumerate(split_string):
        if word in stop_word_set:
            pass
        else:
            if idx == len(split_string)-1:
                return_string = return_string  + word
            else:
                return_string = return_string  + word + ' '
    return return_string
    
def preprocess(data:pd.DataFrame,
               stop_word_set:set,
               save_path:str):
    '''
    前處理
    '''
    max_review_len = 0
    mean_review_len = 0
    review_len_seperate = {}
    category2num = {}

    preprocess_review = []
    preprocess_category = []
    counter = 0
    category2num = {}
    #label轉數字
    try:
        with open(f'{CURRENT_PATH}/category2num.txt') as fp:
            all = fp.readlines()
            for i in all:
                i = i.strip('\n')
                split = i.split(':')
                category2num.setdefault(split[0],split[1])
    except:
        pass

    for idx,(review,sentiment) in tqdm.tqdm(enumerate(zip(data['review'],data['sentiment']))):
        review = seg_sentence(review.lower())
        review = emoji.demojize(review)
        review = rm_stop_word(stop_word_set,review)
        #紀錄最長的review
        if len(review.split()) > max_review_len:
            max_review_len = len(review.split())
        #紀錄長度分布    
        if len(review.split()) in review_len_seperate:
            review_len_seperate[len(review.split())] += 1
            mean_review_len+=len(review.split())
        else:
            review_len_seperate.setdefault(len(review.split()),1)
            mean_review_len+=len(review.split())
        
        #轉成數字標籤
        preprocess_category.append(category2num[sentiment])

        preprocess_review.append(review)
        
    data['review'] = preprocess_review
    data['sentiment'] = preprocess_category
    data.to_csv(save_path,index=False)

    return data, max_review_len, mean_review_len, review_len_seperate, category2num
    
def test(test_loader,
            model,
            ):
    '''
    Test驗證
    '''
    test_avg_acc = 0
    total_acc = 0
    predict = np.array([])
    ans = np.array([])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()    #預測要把model變成eval狀態
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.type(torch.LongTensor)#CE LOSS
            label = label.to(device)
            out = model(data)
            test_p = out.argmax(dim=1) #CE LOSS  
            num_correct = (test_p==label).sum().item() #該batch在test時預測成功的數量   
            batch_acc  = num_correct / label.size(0)
            total_acc += batch_acc

            #把所有預測和答案計算
            predict = np.append(predict,test_p.cpu().numpy())
            ans = np.append(ans,label.cpu().numpy())

    
        test_avg_acc = round(total_acc/len(test_loader),4)
    return test_avg_acc,predict.astype(int),ans.astype(int) 


if __name__ == "__main__":
    CURRENT_PATH = os.path.dirname(__file__)
    GLOVE_PATH = f'{CURRENT_PATH}/glove.6B.100d.txt' 
    STOP_WORD_PATH = f'{CURRENT_PATH}/stop_words_english.txt'
    PREPOCESS_TEST_PATH = f'{CURRENT_PATH}/data/preprocess_test.csv'
    TMP_ROOT = f'/weights/GRU2/'
    SAVE_MODELS_PATH = f'{CURRENT_PATH}/weights/GRU2/epoch_14_trainLoss_0.2613_trainAcc_89.01_valLoss_0.2388_valAcc_90.37.pth'
    seq_len = 180
    hidden_size = 128
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    MODEL = GRU_IMDB(hidden_size=hidden_size,
                     embedding_dim=100,
                     num_classes=2,
                     num_layers=2).to(device=device)
    MODEL.load_state_dict(torch.load(SAVE_MODELS_PATH))
    EPOCH = 1000
    BATCH_SIZE = 512
    LOSS = nn.CrossEntropyLoss()
    LEARNING_RATE = 0.001
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

    if os.path.exists(PREPOCESS_TEST_PATH):
        print("skip preprocess ...")
        test_data = pd.read_csv(PREPOCESS_TEST_PATH)
    else:
        print("prepocess now ...")
        #stop word的set
        stop_word_set=set()
        with open(STOP_WORD_PATH,'r',encoding="utf-8") as fp:
            all = fp.readlines()
            for word in all:
                stop_word_set.add(word.strip('\n'))

        test_csv = pd.read_csv(f'{CURRENT_PATH}/data/test.csv')
        test_data,max_review_len, mean_review_len, review_len_seperate, category2num = preprocess(data = test_csv,
                                                                                    stop_word_set = stop_word_set,
                                                                                    save_path = PREPOCESS_TEST_PATH,)
        print('review_len 最大:',max_review_len)
        print('review 數量分布:',sorted(review_len_seperate.items(), key=lambda item: item[1]))
        print('review 平均數:',(mean_review_len//len(test_csv))+1)#平均數
        print('review 中位數:',statistics.median(review_len_seperate))#中位數
    
    w2v_dic,w2v_mean = w2v_dict(GLOVE_PATH=GLOVE_PATH)
    print("w2v ...")
    tokenize_string = []
    labels = []
    
    for string,label in tqdm.tqdm(zip(test_data['review'],test_data['sentiment'])):
        tokenize = replace_w2v(w2v_dicts=w2v_dic,
                                w2v_dicts_mean=w2v_mean,
                                seg_len=seq_len,
                                string = string,
                                )
        
        torch_tokenize = torch.from_numpy(np.array(tokenize)).float()
        torch_label = torch.from_numpy(np.array(label))

        tokenize_string.append(torch_tokenize)
        labels.append(torch_label)
    
    tokenize_string = np.array(tokenize_string)
    labels = np.array(labels)
    test_data = [tokenize_string,labels]

    test_data = dataset(data=test_data)
    test_dataloader = DataLoader(test_data,batch_size=BATCH_SIZE)

    
    print("testing ...")
    test_acc,predict,ans = test(
        test_loader=test_dataloader,
        model=MODEL,
    )
    print('Test acc:',test_acc)
   
    plt.rcParams['figure.figsize'] = [10, 10]
    disp = ConfusionMatrixDisplay.from_predictions(
        ans,
        predict,
        display_labels=['negative','positive'],
        cmap=plt.cm.Blues,
        normalize='true',
    )
    plt.savefig(f'{CURRENT_PATH}/{TMP_ROOT}/ConfusionMatrix.jpg',bbox_inches='tight')
    