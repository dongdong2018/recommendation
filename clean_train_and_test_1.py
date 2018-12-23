import pandas as pd
import gc
import os

import static_value

import matplotlib.pyplot as plt
import seaborn as sns
#
# members = pd.read_csv('./data/music3_members_1.csv', low_memory=False)
# songs = pd.read_csv('./data/music3_songs_and_songsextra_1.csv', low_memory=False)
#
# test = pd.read_csv('./data/test.csv', low_memory=False)
train = pd.read_csv('./data/train.csv', low_memory=False)

train['weight'] = (train['target'] == 1 ).rolling(window=10000, center=True).mean()
train['weight'].to_csv('liner_weight.csv')
exit()
import numpy as np
for x in train['weight']:
    c= c +1
    print(x)
    if x is np.nan:
        print(c)
        exit()
    exit()
exit()



# train = shuffle(train)
del members['Unnamed: 0']
del songs['Unnamed: 0']
test['target'] = test['id']
del test['id']
print('train length is:', len(train))
print('test length is:', len(test))
print('songs length is:', len(songs))
print('members length is:', len(members))
train_all = pd.concat([train,test], axis=0)
del test
del train
gc.collect()
l = len(train_all)
print('test and train contact length is:', l)
train_all = train_all.fillna(0)
for m in train_all.columns:
    print(m, len(train_all[train_all[m].isnull()])/l,
          len(train_all[train_all[m].isnull()]))


train_all = pd.merge(left=train_all,right=members,
                        how='left', on='msno')
train_all = pd.merge(left=train_all,right=songs,
                        how='left', on='song_id')
del songs
del members
gc.collect()
print(len(train_all))
train_all = train_all.fillna(0)
for m in train_all.columns:
    print(m,len(train_all[train_all[m].isnull()])/l,
          len(train_all[train_all[m].isnull()]))


train_all['RY'] = train_all['regi_years']-train_all['YY']
train_all = train_all.drop(['regi_years', 'YY', 'CC'], axis=1)


d = train_all['source_screen_name'].value_counts()
train_all['source_screen_name'] = d[train_all['source_screen_name'].values].values
d = train_all['source_type'].value_counts()
train_all['source_type'] = d[train_all['source_type'].values].values


d = train_all['language'].value_counts()
train_all['language'] = d[train_all['language'].values].values
d = train_all['artist_name'].value_counts()
train_all['artist_name'] = d[train_all['artist_name'].values].values

d = train_all['name'].value_counts()
train_all['name'] = d[train_all['name'].values].values
d = train_all['XXX'].value_counts()
train_all['XXX'] = d[train_all['XXX'].values].values
d = train_all['NNNNN'].value_counts()
train_all['NNNNN'] =d[train_all['NNNNN'].values].values
d = train_all['msno'].value_counts()
train_all['msno'] = d[train_all['msno'].values].values
d = train_all['song_id'].value_counts()
train_all['song_id'] = d[train_all['song_id'].values].values


train_all['last_col'] = train_all['target']
del train_all['target']
train = train_all[:static_value.train_LEN]
test  = train_all[static_value.train_LEN:]
del train_all
gc.collect()


train['weight'] = (train['last_col'] == 1 ).rolling(window=10000, center=True).mean()
print(train['weight'][:10])



print('after len train test rows:', len(train), len(test))
print('after len train test columns', len(train.columns), len(test.columns))
print(train.head())
print(test.head())
train_file = './data/music3_train_1_noshuffle.csv'
test_file = './data/music3_test_1_noshuffle.csv'
try:
    os.remove(train_file)
    os.remove(test_file)
except:
    pass
print('to csv train data length is:', len(train))
print('to csv test data length is:', len(test))
train.to_csv(train_file)
test.to_csv(test_file)


