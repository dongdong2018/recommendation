import pandas as pd
import gc
import os
import numpy as np
songs_extra = pd.read_csv('./data/song_extra_info.csv')

l = len(songs_extra)
print('the songs_extra length is:', l)
songs_extra = songs_extra[songs_extra['name'].isnull() == False]
songs_extra = songs_extra[songs_extra['isrc'].isnull() == False]
for m in songs_extra.columns:
    print(m, len(songs_extra[songs_extra[m].isnull()])/l,
          len(songs_extra[songs_extra[m].isnull()]))

songs_extra['CC'] = songs_extra['isrc'].str[:2]
songs_extra['XXX'] = songs_extra['isrc'].str[2:5]
songs_extra['YY'] = songs_extra['isrc'].str[5:7]


def YY(x):
    if x >19:
        return 1900+x
    else:
        return 2000+x
songs_extra['YY'] = songs_extra['YY'].map(lambda x: YY(int(x)))
songs_extra['NNNNN'] = songs_extra['isrc'].str[7:]
for m in songs_extra.columns:
    print(m, len(songs_extra[songs_extra[m].isnull()])/l,
          len(songs_extra[songs_extra[m].isnull()]))
del songs_extra['isrc']

print(songs_extra.head())
print(len(songs_extra))
file = './data/music3_songs_extra_1.csv'
songs_extra.to_csv(file)
songs = pd.read_csv(file)
print(len(songs))

