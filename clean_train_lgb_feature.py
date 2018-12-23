import pandas as pd
import static_value
import gc
import lightgbm as lgb
import datetime
from sklearn.model_selection import train_test_split

train_io = pd.read_csv('./data/music3_train_1.csv',
                    dtype={
                           'genre_ids': 'category',
                            'gender': 'category',
                           'source_screen_name' : 'category',
                           'source_system_tab' : 'category',
                           'source_type': 'category',
                            'language': 'category',
                            'bd': 'category',
                           'city' : 'category',
                           'rigi_weekday' : 'category',
                           'expi_weekday' : 'category',
                           'registered_via' : 'category',
                           'regi_years': 'category',
                            'regi_months': 'category',
                             'expi_months': 'category'
                    })
del train_io['Unnamed: 0']
train_data = train_io
y = train_data['last_col'].values
X = train_data

del X['last_col']
del train_io
gc.collect()


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'application': 'regression',
    'is_unbalance': 'true',
    'zero_as_missing': 'true',
    'metric': 'auc',
    'lambda_l2': 0.3,
    'learning_rate': 0.5,
    'cat_smooth': 30,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 15,
    'num_leaves': 7,
    'save_binary':'true',
    'verbose': 0
}


start_time = datetime.datetime.now()
s = datetime.datetime.now()
train_X, valid_X, train_y, valid_y = train_test_split(
    X, y,test_size=static_value.proportionment,
    random_state=True
)
lgb_train = lgb.Dataset(train_X, train_y, free_raw_data=False)
lgb_valid = lgb.Dataset(valid_X, valid_y, free_raw_data=False)
print('start training.....')
evals_result = {}

gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=30,
        evals_result=evals_result,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=5

    )

del lgb_train
del lgb_valid
gc.collect()

y_pre = gbm.predict(X, pred_leaf=True,
                    num_iteration=gbm.best_iteration)
print(len(y_pre), len(y))
p = pd.DataFrame(y_pre)
p['target'] = y
p['msno'] = X['msno']
p['song_id'] = X['song_id']
p['ship_days'] = X['ship_days']
p['song_length'] = X['song_length']
p['name'] = X['name']
p['XXX'] = X['XXX']
p['name'] = X['name']
p['YY'] = X['YY']
p['NNNNN'] = X['NNNNN']
p.to_csv('train3.csv', index=False)

del p
del X
del y
gc.collect()

exit()
test_x = pd.read_csv('./data/music3_test_1.csv',
                    dtype={
                           'genre_ids': 'category',
                            'gender': 'category',
                           'source_screen_name' : 'category',
                           'source_system_tab' : 'category',
                           'source_type': 'category',
                            'language': 'category',
                            'bd': 'category',
                           'city' : 'category',
                           'rigi_weekday' : 'category',
                           'expi_weekday' : 'category',
                           'registered_via' : 'category',
                           'regi_years': 'category',
                            'regi_months': 'category',
                             'expi_months': 'category'
                    })
del test_x['Unnamed: 0']

y = test_x['last_col'].values
X = test_x

del X['last_col']
del test_x
gc.collect()
y_pre = gbm.predict(X, pred_leaf=True,
                    num_iteration=gbm.best_iteration)
print(len(y_pre), len(y))
p = pd.DataFrame(y_pre)
p['target'] = y
p['msno'] = X['msno']
p['song_id'] = X['song_id']
p['ship_days'] = X['ship_days']
p['song_length'] = X['song_length']
p['name'] = X['name']
p['CC'] = X['CC']
p['XXX'] = X['XXX']
p['name'] = X['name']
p['NNNNN'] = X['NNNNN']
p.to_csv('test3.csv', index=False)
print('it is time :', datetime.datetime.now()-s)


