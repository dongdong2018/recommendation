
import pandas as pd
import static_value
import gc
import lightgbm as lgb
import datetime
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

train_io = pd.read_csv('./data/music3_train_1_noshuffle.csv',
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
print(train_io.info())
print('the train data length:', len(train_io))
test_data = train_io[-500000:]
train_data = train_io[:-500000]

y_test = test_data['last_col'].values
x_test = test_data

y = train_data['last_col'].values
X = train_data



del X['last_col']
del x_test['last_col']
del train_io
gc.collect()

valid_X = X[-800000:]
valid_y = y[-800000:]


evals_result_1 = {}
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'application': 'binary',
    'is_unbalance': 'true',
    'min_data_in_leaf': 60,
    'zero_as_missing': 'true',
    'metric': 'auc',
    'lambda_l2': 0.4,
    'learning_rate': 0.05,
    'cat_smooth': 150,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.5,
    'bagging_freq': 31,
    'num_leaves': 5,
    'save_binary':'true',
    'verbose': 0
}

start_time = datetime.datetime.now()

s = datetime.datetime.now()
lgb_train = lgb.Dataset(X, y, free_raw_data=False)
lgb_valid = lgb.Dataset(valid_X, valid_y, free_raw_data=False)

print('start training.....')
evals_result = {}
gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=5,
        evals_result=evals_result,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=2
    )
y_pre = gbm.predict(x_test, pred_leaf=True, num_iteration=gbm.best_iteration)
print(x_test.head())

p = pd.DataFrame(y_pre)
print(len(p), len(x_test))
X = pd.concat([p, x_test], axis=1, ignore_index=True)
print(X.head())
print(len(X))
exit()
lgb_train1 = lgb.Dataset(X, y_test, free_raw_data=False)
gbm = lgb.train(
        params,
        lgb_train1,
        num_boost_round=15,
        evals_result=evals_result,
        valid_sets=[lgb_train1],
        early_stopping_rounds=5
    )
exit()
auc2 = metrics.roc_auc_score(y_test, y_pre)
ax = lgb.plot_metric(evals_result)
plt.show()
ax = lgb.plot_importance(gbm, max_num_features=23)
plt.show()
print('predict is:', auc2)
print('it is time :', datetime.datetime.now()-s)
gbm.save_model(str(auc2)+'.txt')



