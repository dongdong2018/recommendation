import pandas as pd
import torch.utils.data as data
import numpy as np
import static_value

train_io = pd.read_csv('./data/music3_train_3.csv')


train_io = train_io.drop(['gender','regi_months','expi_weekday','expi_months',
                          'genre_ids','language'], axis=1)
train_io = pd.get_dummies(train_io,columns=['regi_weekday'])

# train_io = train_io[:2000000]
print(train_io.info())
train_y = train_io['last_col'].values
del train_io['last_col']
train_X = train_io.values
print(len(train_X), print(len(train_y)))


class Music_data(data.Dataset):
    def __init__(self, train=True,transform=None,
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set


        if self.train:
            self.train_data, self.train_label = train_X[:-static_value.test_LEN+1000000], \
                                                train_y[:-static_value.test_LEN+1000000]
        else:
            self.test_data, self.test_label = train_X[-static_value.test_LEN+1500000:], \
                                                  train_y[-static_value.test_LEN+1500000:]
    def __getitem__(self, index):
        if self.train:
            d1, target = self.train_data[index], self.train_label[index]
        else:
            d1, target = self.test_data[index], self.test_label[index]
        if self.transform is not None:
            d1 = self.transform(d1)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return d1, target
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

print(len(Music_data()))