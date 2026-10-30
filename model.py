import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb, time
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y = train['satisfaction'].astype(int).values
cols = [c for c in train.columns if c not in ['id', 'satisfaction']]
cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
smooth = 5
prior = y.mean()

def key(df, kc):
    return df[kc].astype(str).agg('_'.join, axis=1).values

def encode(key_fit, y_fit, key_apply):
    s = pd.DataFrame({'k': key_fit, 'y': y_fit}).groupby('k')['y'].agg(['sum', 'count'])
    e = (s['sum'] + prior * smooth) / (s['count'] + smooth)
    return pd.Series(key_apply).map(e).fillna(prior).values

A = train[cols].copy()
T = test[cols].copy()
ksets = {'te_fd': ['Flight Distance'], 'te_fdct': ['Flight Distance', 'Class', 'Type of Travel']}
for name, kc in ksets.items():
    kA, kT = key(A, kc), key(T, kc)
    oof = np.zeros(len(A))
    for i, j in KFold(5, shuffle=True, random_state=1).split(A):
        oof[j] = encode(kA[i], y[i], kA[j])
    A[name] = oof
    T[name] = encode(kA, y, kT)
for d in (A, T):
    for c in cat_cols:
        d[c] = pd.Categorical(d[c], categories=sorted(train[c].unique()))

preds = {}

t = time.time()
lgb_preds = []
cfgs = [dict(learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, colsample_bytree=0.6, reg_lambda=5, max_bin=4000, n_estimators=2100),
        dict(learning_rate=0.03, num_leaves=127, min_child_samples=60, subsample=0.7, colsample_bytree=0.5, reg_lambda=10, max_bin=4000, n_estimators=800)]
for i, p in enumerate(cfgs):
    m = lgb.LGBMClassifier(subsample_freq=1, n_jobs=-1, verbose=-1, random_state=i, **p)
    m.fit(A, y)
    lgb_preds.append(m.predict_proba(T)[:, 1])
    print('lgb', i, 'done %.0fs' % (time.time() - t), flush=True)
preds['lgb'] = np.mean(lgb_preds, axis=0)
np.save('full_lgb.npy', preds['lgb'])

t = time.time()
mx = xgb.XGBClassifier(n_estimators=950, learning_rate=0.06, max_depth=7, min_child_weight=5, subsample=0.8,
                       colsample_bytree=0.6, reg_lambda=5, tree_method='hist', max_bin=1024,
                       enable_categorical=True, n_jobs=-1)
mx.fit(A, y)
preds['xgb'] = mx.predict_proba(T)[:, 1]
np.save('full_xgb.npy', preds['xgb'])
print('xgb done %.0fs' % (time.time() - t), flush=True)

t = time.time()
XC = train[cols].copy()
TC = test[cols].copy()
for d in (XC, TC):
    d['fd_cat'] = d['Flight Distance'].astype(str)
    d['fd_class_travel'] = d['Flight Distance'].astype(str) + '_' + d['Class'] + '_' + d['Type of Travel']
cats = cat_cols + ['fd_cat', 'fd_class_travel']
mc = CatBoostClassifier(iterations=950, learning_rate=0.1, depth=8, thread_count=-1, verbose=200, random_seed=0)
mc.fit(Pool(XC, y, cat_features=cats))
preds['cat'] = mc.predict_proba(Pool(TC, cat_features=cats))[:, 1]
np.save('full_cat.npy', preds['cat'])
print('cat done %.0fs' % (time.time() - t), flush=True)

print(pd.DataFrame(preds).corr(method='spearman'))
blend = (preds['lgb'] + preds['xgb'] + preds['cat']) / 3
pd.DataFrame({'id': test['id'], 'satisfaction': blend}).to_csv('submission.csv', index=False)
print('done')