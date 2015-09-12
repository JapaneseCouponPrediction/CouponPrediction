# -*- coding: utf-8 -*-
"""
Kaggle coupon prediction

Created on Fri Aug 28 16:27:01 2015

@author: Yujie
"""
import os
import numpy as np
import pandas as pd
import getpass

user = getpass.getuser()
baseDir = 'C:\Users\\' + user + '\\Google Drive\\datascientist\\projects\\myKaggle\\CouponPrediction'
#%% use the provided documentation to translate
# build dictionary for translation
df = pd.read_excel(os.path.join(baseDir,'data','documentation','CAPSULE_TEXT_Translation.xlsx'),skiprows=5)
k = [x for x in df['CAPSULE_TEXT']]
v = [x for x in df['English Translation']]
capsule_transmap = dict(zip(k,v))

k = [x for x in df['CAPSULE_TEXT.1']]
v = [x for x in df['English Translation.1']]
genre_transmap = dict(zip(k,v))

# translate csv
f = pd.read_csv(os.path.join(baseDir,'data','coupon_list_train.csv'),encoding='utf-8')
a = [capsule_transmap[x] for x in f['CAPSULE_TEXT']]
f['CAPSULE_TEXT'] = [capsule_transmap[x] for x in df['CAPSULE_TEXT']] 
#--- this leaves area name unchanged, use json to translate all

#%% use json dict to translate
execfile(os.path.join(baseDir, 'json_trans.py'))
trans_map[np.nan] = np.nan # add np.nan to take care of missing value
colnamelist = ['CAPSULE_TEXT','PREF_NAME','GENRE_NAME','KEN_NAME',
           'SMALL_AREA_NAME','PREFECTUAL_OFFICE','LARGE_AREA_NAME',
           'small_area_name','ken_name','large_area_name']
for f in ['coupon_area_test','coupon_area_train',
          'coupon_list_test','coupon_list_train',
          'coupon_detail_train',
          'user_list',
          'prefecture_locations']:
    df = pd.read_csv(baseDir+'\\data\\'+f+'.csv', encoding='utf-8')
    df.columns = [str(c) for c in df.columns]
    for colname in colnamelist:
        if colname in df.columns:
            df[colname] = [trans_map[v] for v in df[colname]]
        else:
            print '%s is not in %s'%(colname,f)
    df.to_csv(baseDir+'\\data\\'+f+'_translated.csv',index=False)
        
    
#%% content-based
# join coupon_area and coupon_list
cpn_list_train = pd.read_csv(os.path.join(baseDir,'data','coupon_list_train_translated.csv'),
                             encoding='utf-8')
cpn_area_train = pd.read_csv(os.path.join(baseDir,'data','coupon_area_train_translated.csv'),
                             encoding='utf-8')
cpn_listarea_train = cpn_list_train.join(cpn_area_train, on='COUPON_ID_HASH',how='left',
                                         lsuffix='_list',rsuffix='_area')
user_list = pd.read_csv(os.path.join(baseDir,'data','user_list_translated.csv'),
                             encoding='utf-8')

# create OHE features
from sklearn.feature_extraction import DictVectorizer
cpn_list_train_rawfeat = cpn_list_train[['GENRE_NAME','CAPSULE_TEXT']].to_dict('records')
vec = DictVectorizer()
cpn_list_train_OHE = vec.fit_transform(cpn_list_train_rawfeat).toarray()
cpn_detail_train = pd.read_csv(os.path.join(baseDir,'data','coupon_detail_train_translated.csv'),
                               encoding='utf-8')

def get_user_OHE(uid):
    ''' get user OHE feature. sum-up coupon's user purchased (by row), 
    and normalize (by row)
        will add area etc. info later
    '''
    buy_cpn = cpn_detail_train[cpn_detail_train['USER_ID_HASH']==uid].COUPON_ID_HASH.unique()                
    prof = cpn_list_train_OHE[cpn_list_train['COUPON_ID_HASH'].isin(list(buy_cpn)).values,:]
    prof_sum = np.nansum(prof,axis=0)
    user_OHE = prof_sum
    return user_OHE
user_OHE = map(lambda x: get_user_OHE(x), user_list.USER_ID_HASH[:10])
# calculate similarity bw user profile and item profile
import sklearn.preprocessing as preprocessing
cpn_list_train_OHE_norm = preprocessing.normalize(cpn_list_train_OHE, norm='l2')
user_OHE_norm = preprocessing.normalize(user_OHE, norm='l2')

# make prediction
from sklearn.metrics.pairwise import linear_kernel
gram = linear_kernel(user_OHE_norm, cpn_list_train_OHE_norm)
