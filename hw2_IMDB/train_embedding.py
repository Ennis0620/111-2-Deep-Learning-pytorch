import os,shutil,re
import pandas as pd
import emoji,tqdm
import torch
import statistics
import numpy as np
import torch.nn as nn
from early_stop import early_stop
from model import RNN_IMDB_EMBEDDING
import matplotlib.pyplot as plt
from dataset_train import dataset
from torch.utils.data import DataLoader

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
    no_mapping = 0
    for word in string_split:
        #print('per word:',word)
        if word in w2v_dicts:
            return_list.append(w2v_dicts[word])
        else:
            return_list.append(w2v_dicts_mean)
            no_mapping += 1
            print(no_mapping)
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
        if sentiment not in category2num:
            category2num.setdefault(sentiment,counter)
            preprocess_category.append(counter)
            counter += 1
        else:
            preprocess_category.append(category2num[sentiment])

        preprocess_review.append(review)
        
    data['review'] = preprocess_review
    data['sentiment'] = preprocess_category
    data.to_csv(save_path,index=False)

    return data, max_review_len, mean_review_len, review_len_seperate, category2num
    
def train(train_loader,
        valid_loader,
        model,
        optimizer,
        loss_func,
        epoch, 
        ):
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for num_epoch in range(epoch):
        train_avg_loss = 0                #每個epoch的loss
        train_avg_acc = 0                 #每個epoch的acc
        total_acc = 0
        for data,label in tqdm.tqdm(train_loader):
            #確保每一batch都能進入model.train模式
            model.train()
            #放置gpu訓練            
            data = data.to(device)
            label = label.type(torch.LongTensor)
            label = label.to(device)
            out = model(data)
            loss = loss_func(out,label)
            optimizer.zero_grad()
            loss.backward()
            #更新優化器
            optimizer.step()
            #累加每個batch的loss後續再除step數量
            train_avg_loss += loss.item()
            #計算acc
            train_p = out.argmax(dim=1)                 #取得預測的最大值
            num_correct = (train_p==label).sum().item() #該batch在train時預測成功的數量
            batch_acc  = num_correct / label.size(0)
            total_acc += batch_acc
        
        val_avg_loss,val_avg_acc = valid(
            valid_loader=valid_loader,
            model=model,
            loss_func=loss_func
        )    

        train_avg_loss = round(train_avg_loss/len(train_loader),4)   #該epoch每個batch累加的loss平均
        train_avg_acc = round(total_acc/len(train_loader),4)         #該epoch的acc平均

        train_loss.append(train_avg_loss)
        train_acc.append(train_avg_acc)
        valid_loss.append(val_avg_loss)
        valid_acc.append(val_avg_acc)

        print('Epoch: {} | train_loss: {} | train_acc: {}% | val_loss: {} | val_acc: {}%'\
              .format(num_epoch, train_avg_loss,round(train_avg_acc*100,4),val_avg_loss,round(val_avg_acc*100,4)))
        
        performance_value = [num_epoch,
                                train_avg_loss,
                                round(train_avg_acc*100,4),
                                val_avg_loss,
                                round(val_avg_acc*100,4)]
        EARLY_STOP(val_avg_acc,
                    model=model,
                    performance_value = performance_value
                    )
            
        if EARLY_STOP.early_stop:
            print('Early stopping')
            break   
    return train_loss,train_acc,valid_loss,valid_acc 

def valid(valid_loader,
            model,
            loss_func):
    '''
    訓練時的驗證
    '''
    val_avg_loss = 0
    val_avg_acc = 0
    total_acc = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()    #預測要把model變成eval狀態
    with torch.no_grad():
        for data, label in valid_loader:
            data = data.to(device)
            label = label.type(torch.LongTensor)
            label = label.to(device)
            out = model(data)
            loss = loss_func(out,label)
            #累加每個batch的loss後續再除step數量
            val_avg_loss += loss.item()
            valid_p = out.argmax(dim=1)   
            num_correct = (valid_p==label).sum().item() #該batch在train時預測成功的數量   
            batch_acc  = num_correct / label.size(0)
            total_acc += batch_acc
    
        val_avg_loss = round(val_avg_loss/len(valid_loader),4)
        val_avg_acc = round(total_acc/len(valid_loader),4)
    return val_avg_loss,val_avg_acc

def plot_statistics(train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                    SAVE_MODELS_PATH):
    '''
    統計train、valid的loss、acc
    '''
    
    t_loss = plt.plot(train_loss)
    t_acc = plt.plot(train_acc)
    v_loss = plt.plot(valid_loss)
    v_acc = plt.plot(valid_acc)
    
    plt.legend([t_loss,t_acc,v_loss,v_acc],
               labels=['train_loss',
                        'train_acc',
                        'valid_loss',
                        'valid_acc'])
    plt.savefig(f'{SAVE_MODELS_PATH}/train_statistics',bbox_inches='tight')
    plt.figure()


if __name__ == "__main__":
    CURRENT_PATH = os.path.dirname(__file__)
    GLOVE_PATH = f'{CURRENT_PATH}/glove.6B.100d.txt' 
    STOP_WORD_PATH = f'{CURRENT_PATH}/stop_words_english.txt'
    PREPOCESS_TRAIN_PATH = f'{CURRENT_PATH}/data/preprocess_train.csv'
    
    SAVE_MODELS_PATH = f'{CURRENT_PATH}/weights/rnn'
    seq_len = 130
    hidden_size = 64
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    MODEL =RNN_IMDB_EMBEDDING(vocab_size=seq_len,
                            hidden_size=hidden_size,
                            embedding_dim=100,
                            num_classes=2).to(device=device)
    EPOCH = 1000
    BATCH_SIZE = 512
    LOSS = nn.CrossEntropyLoss()
    LEARNING_RATE = 0.001
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

    EARLY_STOP = early_stop(save_path=SAVE_MODELS_PATH,
                            mode='max',
                            monitor='val_acc',
                            patience=15)
    
    
    if os.path.exists(PREPOCESS_TRAIN_PATH):
        print("skip preprocess ...")
        train_data = pd.read_csv(PREPOCESS_TRAIN_PATH)
    else:
        print("prepocess now ...")
        #stop word的set
        stop_word_set=set()
        with open(STOP_WORD_PATH,'r',encoding="utf-8") as fp:
            all = fp.readlines()
            for word in all:
                stop_word_set.add(word.strip('\n'))

        train_csv = pd.read_csv(f'{CURRENT_PATH}/data/train.csv')
        train_data,max_review_len, mean_review_len, review_len_seperate, category2num = preprocess(data = train_csv,
                                                                                    stop_word_set = stop_word_set,
                                                                                    save_path = PREPOCESS_TRAIN_PATH,)
        print('review_len 最大:',max_review_len)
        print('review 數量分布:',sorted(review_len_seperate.items(), key=lambda item: item[1]))
        print('review 平均數:',(mean_review_len//len(train_csv))+1)#平均數
        print('review 中位數:',statistics.median(review_len_seperate))#中位數
        
        try:
            os.remove(f'{CURRENT_PATH}/category2num.txt')
        except:
            pass
        with open(f'{CURRENT_PATH}/category2num.txt','a+') as fp:
            for key in category2num:
                fp.write(key + ":" + str(category2num[key]) +'\n')

    
    w2v_dic,w2v_mean = w2v_dict(GLOVE_PATH=GLOVE_PATH)
    print("w2v ...")
    tokenize_string = []
    labels = []
    
    for string,label in tqdm.tqdm(zip(train_data['review'],train_data['sentiment'])):
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

    print("suffule data ...")
    #正負評各2萬筆
    pos_indices = np.where(labels==1)
    neg_indices = np.where(labels==0)
    pos_indices = pos_indices[0]
    neg_indices = neg_indices[0]
    
    #train:正評:17500 負評:17500, valid:正評:2500 負評:2500
    split = (1/8)
    tmp_split = int(np.floor(split * len(pos_indices)))
    train_pos_indices,valid_pos_indices = pos_indices[tmp_split:], pos_indices[:tmp_split]
    train_neg_indices,valid_neg_indices = neg_indices[tmp_split:], neg_indices[:tmp_split]

    new_train_tokenize_string = []
    new_train_labels = []
    new_valid_tokenize_string = []
    new_valid_labels = []

    for p,n in zip(train_pos_indices,train_neg_indices):
        new_train_tokenize_string.append(tokenize_string[p])
        new_train_tokenize_string.append(tokenize_string[n])
        new_train_labels.append(labels[p])
        new_train_labels.append(labels[n])

    for p,n in zip(valid_pos_indices,valid_neg_indices):
        new_valid_tokenize_string.append(tokenize_string[p])
        new_valid_tokenize_string.append(tokenize_string[n])
        new_valid_labels.append(labels[p])
        new_valid_labels.append(labels[n])

    new_train_tokenize_string = np.array(new_train_tokenize_string)
    new_train_labels = np.array(new_train_labels)
    new_valid_tokenize_string = np.array(new_valid_tokenize_string)
    new_valid_labels = np.array(new_valid_labels)
    #打亂index
    train_indices = list(range(len(new_train_tokenize_string)))
    valid_indices = list(range(len(new_valid_tokenize_string)))
    np.random.shuffle(train_indices)
    np.random.shuffle(valid_indices)
    #將train、valid依照index打亂
    new_train_tokenize_string = new_train_tokenize_string[train_indices]
    new_train_labels = new_train_labels[train_indices]
    new_valid_tokenize_string = new_valid_tokenize_string[valid_indices]
    new_valid_labels = new_valid_labels[valid_indices]

    train_data = [new_train_tokenize_string,new_train_labels]
    valid_data = [new_valid_tokenize_string,new_valid_labels]




    train_data = dataset(data=train_data)
    train_dataloader = DataLoader(train_data,batch_size=BATCH_SIZE)

    valid_data = dataset(data=valid_data)
    valid_dataloader = DataLoader(valid_data,batch_size=BATCH_SIZE)
    
    
    print("training ...")
    train_loss,train_acc,valid_loss,valid_acc = train(
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        model=MODEL,
        optimizer=OPTIMIZER,
        loss_func=LOSS,
        epoch=EPOCH
    )
    plot_statistics(train_loss,train_acc,valid_loss,valid_acc,SAVE_MODELS_PATH)
    
    
    