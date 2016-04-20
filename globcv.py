import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


print('Processing...', flush=True)
global_train = pd.read_csv('data/train.csv', index_col='ID')
global_test = pd.read_csv('data/test.csv', index_col='ID')

global_target = global_train.TARGET.values.copy()
global_train.drop('TARGET', axis=1, inplace=True)

global_train.loc[:, 'n0'] = (global_train == 0).sum(axis=1)
global_test.loc[:, 'n0'] = (global_test == 0).sum(axis=1)

print('Cleaning std == 0...', flush=True)
dlist = list()
for c in global_train.columns:
    if global_train[c].unique().shape[0] == 1:
        # print(c)
        dlist.append(c)
global_train.drop(dlist, axis=1, inplace=True)
global_test.drop(dlist, axis=1, inplace=True)

print('Cleaning duplicates...', flush=True)
for cone in global_train.columns:
    for ctwo in global_train.columns:
        if cone not in global_train.columns:
            continue
        if ctwo not in global_train.columns:
            continue
        if cone != ctwo and np.all(global_train[cone] == global_train[ctwo]):
            # print(cone, ctwo)
            global_train.drop(ctwo, axis=1, inplace=True)
            global_test.drop(ctwo, axis=1, inplace=True)

for c in global_train.columns:
    cmin = global_train[c].min()
    cmax = global_train[c].max()
    global_test.loc[global_test[c] < cmin, c] = cmin
    global_test.loc[global_test[c] > cmax, c] = cmax

print('Starting cross validation...', flush=True)

skf = StratifiedKFold(global_target, n_folds=10, random_state=42)

cv_res = list()

for train_index, test_index in tqdm(skf):
    train, test = global_train.iloc[train_index], global_train.iloc[test_index]
    target, y_test = global_target[train_index], global_target[test_index]

    dtrain = xgb.DMatrix(train, target)
    params = {'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'eta': 0.0202048,
              'max_depth': 5,
              'subsample': 0.6815,
              'colsample_bytree': 0.701,
              'silent': 1,
              'seed': 0
              }
    dtest = xgb.DMatrix(test)
    gbm = xgb.train(params, dtrain, num_boost_round=560)
    preds = gbm.predict(dtest)

    cv_res.append(roc_auc_score(y_test, preds))

print('CV results: ROC AUC = {:.6f} +/- {:.6f}'.format(np.mean(cv_res),
                                                       np.std(cv_res)))
