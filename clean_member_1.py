import pandas as pd
import numpy as np

data_members = pd.read_csv('./data/members.csv',
                           dtype={'city' : 'category',
                            'bd' : np.uint8,
                              'registered_via' : 'category'},
                     parse_dates=['registration_init_time',
                                  'expiration_date'])
print(data_members.head())
print(data_members.describe())
print(data_members['bd'].unique())
print(len(data_members[data_members['gender'] == 'female']))
print(len(data_members[data_members['gender'] == 'male']))
data_members['gender'] = data_members['gender'].fillna(0)
for m in data_members.columns:
    print(m, len(data_members[data_members[m].isnull()]))

def fmember_age(x):
    if x <14 or x >57:
        return 0
    else:
        return x
def fmember_sex(x):
    if x == 'female':
        return 1
    elif x == 'male':
        return 2
    else:
        return 0

data_members['regi_years'] = data_members['registration_init_time'].dt.year
data_members['regi_months'] = data_members['registration_init_time'].dt.month
data_members['regi_weekday'] = data_members['registration_init_time'].dt.weekday
data_members['expi_weekday'] = data_members['expiration_date'].dt.weekday
data_members['expi_months'] = data_members['expiration_date'].dt.month
data_members['ship_days'] = data_members['expiration_date'].subtract(
    data_members['registration_init_time']
).dt.days.astype(int)
data_members['bd'] = data_members['bd'].map(lambda x: fmember_age(x))
data_members=data_members.drop(['registration_init_time', 'expiration_date'], axis=1)
data_members.to_csv('./data/music3_members_1.csv')

