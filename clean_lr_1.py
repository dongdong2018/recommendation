import gc
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import datetime
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
s=datetime.datetime.now()

train = pd.read_csv('train3.csv')[:1000000]
print('init features:', train.info())

# train = train.iloc[:, :]

# train = train.iloc[:,30:]
c = train.iloc[:, :-10].columns

print(c)

# print('the one hot features:')
# print(c)
# c = shuffle(c)
# print(c)
# train = train.drop(c[3:], axis=1)
# print('random have features:')
# print(train.head())
train = pd.get_dummies(data=train,
                       columns=c)
print(train.head())
print(train.info())
y = train['target']
del train['target']
X = train
del train
gc.collect()
train_X, valid_X, train_y, valid_y = train_test_split(
    X, y,test_size=0.3,
    random_state=True
)

del X
gc.collect()
lm = LogisticRegression()
lm.fit(train_X, train_y)


y_pre = lm.predict_proba(valid_X)[:, 1]
auc = roc_auc_score(valid_y, y_pre)
print('predict scores is:', auc)
print(datetime.datetime.now() - s)