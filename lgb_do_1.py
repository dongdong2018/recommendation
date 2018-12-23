import pandas as pd
import gc
import lightgbm as lgb
import datetime


filename='11-1.csv.gz'
train = pd.read_csv('./data/music3_test_1_noshuffle.csv',
                    dtype={
                        'genre_ids': 'category',
                        'gender': 'category',
                        'source_screen_name': 'category',
                        'source_system_tab': 'category',
                        'source_type': 'category',
                        'language': 'category',
                        'bd': 'category',
                        'city': 'category',
                        'rigi_weekday': 'category',
                        'expi_weekday': 'category',
                        'registered_via': 'category',
                        'regi_years': 'category',
                        'regi_months': 'category',
                        'expi_months': 'category'
                    })
del train['Unnamed: 0']
y_test = train['last_col'].values
del train['last_col']
x_data = train
del train
gc.collect()

print('start traing.....')
s=datetime.datetime.now()

gbm = lgb.Booster(model_file='0.695860271022.txt')
y_pre = gbm.predict(x_data)
print('load mode is ok!------', datetime.datetime.now() - s)
if max(y_pre)>1 or min(y_pre)<0:
    print('the y_pre are not ill value!')
    exit()
sub = pd.DataFrame()
sub['id'] = y_test
sub['target'] = y_pre
sub.to_csv(filename,
           compression='gzip',
           index=False,
           float_format='%.5f')
print('this do data length is:', len(sub))

print('result finished!')
