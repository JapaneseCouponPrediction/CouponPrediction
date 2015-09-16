# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 19:38:50 2015

@author: Yujie
"""
#%%
import os
import numpy as np
import pandas as pd
import getpass

user = getpass.getuser()
baseDir = 'C:\Users\\' + user + '\\Google Drive\\datascientist\\projects\\myKaggle\\CouponPrediction'


coupon_list_train = pd.read_csv(os.path.join(baseDir,'data','coupon_list_train_translated.csv'),encoding='utf-8')
coupon_list_test = pd.read_csv(os.path.join(baseDir,'data','coupon_list_test_translated.csv'),encoding='utf-8')
user_list = pd.read_csv(os.path.join(baseDir,'data','user_list_translated.csv'))
coupon_purchases_train = pd.read_csv(os.path.join(baseDir,'data','coupon_detail_train_translated.csv'))

### merge to obtain (USER_ID) <-> (COUPON_ID with features) training set
purchased_coupons_train = coupon_purchases_train.merge(coupon_list_train,
                                                 on='COUPON_ID_hash',
                                                 how='inner')

### filter redundant features
features = ['COUPON_ID_hash', 'USER_ID_hash',
            'GENRE_NAME', 'DISCOUNT_PRICE', 'PRICE_RATE',
            'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU',
            'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
            'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name', 'small_area_name']
purchased_coupons_train = purchased_coupons_train[features]

### create 'dummyuser' records in order to merge training and testing sets in one
coupon_list_test['USER_ID_hash'] = 'dummyuser'

### filter testing set consistently with training set
coupon_list_test = coupon_list_test[features]

### merge sets together
combined = pd.concat([purchased_coupons_train, coupon_list_test], axis=0)


### create two new features
combined['DISCOUNT_PRICE'] = 1. / np.log10(combined['DISCOUNT_PRICE'])
combined['PRICE_RATE'] = (combined['PRICE_RATE'] / 100.) ** 2.
features.extend(['DISCOUNT_PRICE', 'PRICE_RATE'])

### convert categoricals to OneHotEncoder form
categoricals = ['GENRE_NAME', 'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED',
                'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN',
                'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name', 'small_area_name']
combined_categoricals = combined[categoricals]
combined_categoricals = pd.concat([pd.get_dummies(combined_categoricals[col],dummy_na=False) 
                                       for col in combined_categoricals.columns], axis=1)
                                        

### leaving continuous features as is, obtain transformed dataset
continuous = list(set(features) - set(categoricals))
combined = pd.concat([combined[continuous], combined_categoricals], axis=1)



### split back into training and testing sets
train = combined[combined['USER_ID_hash'] != 'dummyuser']
test = combined[combined['USER_ID_hash'] == 'dummyuser']
test.drop('USER_ID_hash', inplace=True, axis=1)

### find most appropriate coupon for every user (mean of all purchased coupons), in other words, user profile
train_dropped_coupons = train.drop('COUPON_ID_hash', axis=1)
user_profiles = train_dropped_coupons.groupby(by='USER_ID_hash').apply(np.mean)
#user_profiles['DISCOUNT_PRICE'] = 1
#user_profiles['PRICE_RATE'] = 1

### remove NaN values
NAN_SUBSTITUTION_VALUE = 1
combined = combined.fillna(NAN_SUBSTITUTION_VALUE)

### creating weight matrix for features
FEATURE_WEIGHTS = {
    'GENRE_NAME': 2.05,
    'DISCOUNT_PRICE': 2,
    'PRICE_RATE': -0.13,
    'large_area_name': 0.5,
    'ken_name': 1.01,
    'small_area_name': 4.75
}

# dict lookup helper
def find_appropriate_weight(weights_dict, colname):
    for col, weight in weights_dict.items():
        if colname in coupon_list_test[col].unique() or colname == col:
            return weight
    return 0
    #raise ValueError

W_values = [find_appropriate_weight(FEATURE_WEIGHTS, str(colname))
            for colname in user_profiles.columns]
W = np.diag(W_values)

### find weighted dot product(modified cosine similarity) between each test coupon and user profiles
test_only_features = test.drop('COUPON_ID_hash', axis=1)
similarity_scores = np.dot(np.dot(user_profiles, W),
                           test_only_features.T)

### create (USED_ID)x(COUPON_ID) dataframe, similarity scores as values
coupons_ids = test['COUPON_ID_hash']
index = user_profiles.index
columns = [coupons_ids[i] for i in range(0, similarity_scores.shape[1])]
result_df = pd.DataFrame(index=index, columns=columns,
                      data=similarity_scores)

### obtain string of top10 hashes according to similarity scores for every user
def get_top10_coupon_hashes_string(row):
    row.sort()
    return ' '.join(row.index[-10:][::-1].tolist())

output = result_df.apply(get_top10_coupon_hashes_string, axis=1)


output_df = pd.DataFrame(data={'USER_ID_hash': output.index,
                               'PURCHASED_COUPONS': output.values})
output_df_all_users = pd.merge(user_list, output_df, how='left', on='USER_ID_hash')
output_df_all_users.to_csv(baseDir+'\\cosine_sim_python.csv', header=True,
                           index=False, columns=['USER_ID_hash', 'PURCHASED_COUPONS'])





