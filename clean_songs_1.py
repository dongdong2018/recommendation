import pandas as pd
import datetime

date = datetime.datetime.today()
f1 = './data/songs.csv'
songs = pd.read_csv(f1,dtype={'genre_ids': 'object',
                                                  'language' : 'object',
                                                  'artist_name' : 'object',
                                                  'composer' : 'object',
                                                  'lyricist' : 'category',
                                                  'song_id' : 'category'})
del songs['lyricist']
del songs['composer']
print(len(songs))
l = len(songs)
print(songs.describe())
songs['language'] = songs['language'].fillna(0)
songs['genre_ids'] = songs['genre_ids'].fillna(0)
for m in songs.columns:
    print(m, len(songs[songs[m].isnull()])/l,
          len(songs[songs[m].isnull()]))
songs['song_length'] = songs['song_length'].map(lambda x: int(x/1000))
print(len(songs))
file = './data/music3_songs_1.csv'
songs.to_csv(file)
songs = pd.read_csv(file)
print(len(songs))
