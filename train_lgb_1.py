import pandas as pd
import static_value
import gc
import lightgbm as lgb
import datetime
from sklearn import metrics

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_io = pd.read_csv('./data/music3_train_1.csv',
                    dtype={
                           'genre_ids': 'category',
                            'gender': 'category',
                           'source_system_tab' : 'category',
                            'language': 'category',
                            'bd': 'category',
                           'city' : 'category',
                            'registered_via' : 'category',
                           'rigi_weekday' : 'category',
                           'expi_weekday' : 'category',
                            'regi_months': 'category',
                             'expi_months': 'category'
                    })

del train_io['Unnamed: 0']
print(train_io.info())
print('the train data length:', len(train_io))
test_data = train_io[-static_value.test_data_len:]
train_data = train_io[:-static_value.test_data_len]
print('the test data length:', len(test_data))
print('the train data length:', len(train_data))
y_test = test_data['last_col'].values
x_test = test_data

y = train_data['last_col'].values
X = train_data

del X['last_col']
del x_test['last_col']
del train_io
gc.collect()

evals_result_1 = {}
auc_curve = []
e1 = []
e2 = []
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
    'cat_smooth': 30,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.5,
    'bagging_freq': 15,
    'num_leaves': 1000,
    'save_binary':'true',
    'verbose': 0
}

a = 17
start_time = datetime.datetime.now()
for n in [a,15,91]:
    s = datetime.datetime.now()
    train_X, valid_X, train_y, valid_y = train_test_split(
        X, y,test_size=static_value.proportionment,
        random_state=True
    )
    lgb_train = lgb.Dataset(train_X, train_y, free_raw_data=False)
    lgb_valid = lgb.Dataset(valid_X, valid_y, free_raw_data=False)
    print('start training.....')
    evals_result = {}
    if n == a:
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=400,
            evals_result=evals_result,
            valid_sets=[lgb_train, lgb_valid],
            early_stopping_rounds=10

        )
    else:
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=400,
            evals_result=evals_result,
            valid_sets=[lgb_train, lgb_valid],
            init_model=gbm,
            early_stopping_rounds=10

        )

    y_pre = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    auc2 = metrics.roc_auc_score(y_test, y_pre)
    print('predict is:', auc2)
    print('it is time :', datetime.datetime.now()-s)
    auc_curve.append(auc2)
    e1.append(evals_result['valid_1']['auc'])
    e2.append(evals_result['training']['auc'])
    gbm.save_model(str(n+auc2)+'.txt')
y =auc_curve
auc_train = []
auc_valid = []
for m in e2:
    for m1 in m:
        auc_train.append(m1)
for m in e1:
    for m1 in m:
        auc_valid.append(m1)
elength = range(len(auc_train))


plt.plot(elength, auc_valid, label='valid')
plt.plot(elength, auc_train, label='train')
plt.grid(True)
plt.legend()
plt.show()
ax = lgb.plot_importance(gbm, max_num_features=27)
plt.show()

x = range(len(auc_curve))
plt.plot(x, y, label='auc')
plt.show()





exit()