import pandas as pd
import os 
import numpy as np

CURRENT_PATH = os.path.dirname(__file__)
csv_path = f'{CURRENT_PATH}/data/IMDB Dataset.csv'
IMDB = pd.read_csv(csv_path)
IMDB_sentiment = IMDB['sentiment']

train = {
    "review":[],
    "sentiment":[]
}
test = {
    "review":[],
    "sentiment":[]
}
pos = {
    "review":[],
    "sentiment":[]
}
neg = {
    "review":[],
    "sentiment":[]
}

for review,sentiment in zip(IMDB['review'],IMDB['sentiment']):
    if sentiment == 'positive':
        pos['review'].append(review)
        pos['sentiment'].append(sentiment)
    else:
        neg['review'].append(review)
        neg['sentiment'].append(sentiment)

np.random.seed(2023)
test_split = 0.2
pos_indices = list(range(len(pos['review'])))
np.random.shuffle(pos_indices)    
pos_train_split = int(np.floor(test_split * len(pos['review'])))
pos_train_indices, pos_test_indices = pos_indices[pos_train_split:], pos_indices[:pos_train_split]

neg_indices = list(range(len(neg['review'])))
np.random.shuffle(neg_indices)
neg_train_split = int(np.floor(test_split * len(neg['review'])))
neg_train_indices, neg_test_indices = neg_indices[neg_train_split:], neg_indices[:neg_train_split]

print("pos_train_indices",len(pos_train_indices))
print("pos_test_indices",len(pos_test_indices))

print("neg_train_indices",len(neg_train_indices))
print("neg_test_indices",len(neg_test_indices))


for i in pos_train_indices:
    train['review'].append(pos['review'][i])
    train['sentiment'].append(pos['sentiment'][i])

for i in pos_test_indices:
    test['review'].append(pos['review'][i])
    test['sentiment'].append(pos['sentiment'][i])


for i in neg_train_indices:
    train['review'].append(neg['review'][i])
    train['sentiment'].append(neg['sentiment'][i])

for i in neg_test_indices:
    test['review'].append(neg['review'][i])
    test['sentiment'].append(neg['sentiment'][i])

train = pd.DataFrame(train)
test = pd.DataFrame(test)


train = train.iloc[np.random.permutation(len(train))]
train = train.reset_index(drop=True)

test = test.iloc[np.random.permutation(len(test))]
test = test.reset_index(drop=True)

train.to_csv(f'{CURRENT_PATH}/data/train.csv',index=False)
test.to_csv(f'{CURRENT_PATH}/data/test.csv',index=False)