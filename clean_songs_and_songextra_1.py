import pandas as pd
import gc

songs = pd.read_csv('./data/music3_songs_1.csv', low_memory=False)
songs_extra = pd.read_csv('./data/music3_songs_extra_1.csv')
del songs_extra['Unnamed: 0']
del songs['Unnamed: 0']
print('songs length', len(songs))
print('songs_extra length', len(songs_extra))
songs = pd.merge(left=songs, right=songs_extra,
             how='inner', on='song_id')
del songs_extra
gc.collect()
print('after merge songs length ', len(songs))
file = './data/music3_songs_and_songsextra_1.csv'
songs.to_csv(file)
songs = pd.read_csv(file)
print(len(songs))