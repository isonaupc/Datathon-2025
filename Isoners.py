# Baseline pipeline for Mango-style demand forecasting
# Requirements: pandas, numpy, scikit-learn, lightgbm

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
import lightgbm as lgb

# ---------- 1. Load ----------
train = pd.read_csv('train.csv', low_memory=False)
test  = pd.read_csv('test.csv', low_memory=False)
sample_sub = pd.read_csv('sample_submission.csv')

# ---------- 2. Aggregate target to model-season ----------
# If weekly_demand exists per row: aggregate per id (model-season)
# Some datasets already have multiple rows per id (weeks); we aggregate
agg_target = train.groupby('ID', as_index=False)['weekly_demand'].sum().rename(columns={'weekly_demand':'demand'})
# merge demand into train-level metadata (pick unique per id rows)
meta = train.drop_duplicates(subset=['ID'])  # if each id has many weekly rows, keep one row for static metadata
meta = meta.merge(agg_target, on='ID', how='left')

# For test, we need meta rows for each id in sample_submission (use test.csv unique rows)
test_meta = test.drop_duplicates(subset=['ID']).copy()

# ---------- 3. Parse / extract image embedding vector ----------
# image_embedding often stored as string like "[0.12, 0.34, ...]"
def parse_embedding(s):
    try:
        return np.fromstring(s.strip("[]"), sep=',')
    except:
        return np.array([])

# build matrix for PCA if embeddings exist
if 'image_embedding' in meta.columns and meta['image_embedding'].notnull().any():
    emb_train = np.vstack(meta['image_embedding'].fillna('[]').apply(parse_embedding).values)
    emb_test = np.vstack(test_meta['image_embedding'].fillna('[]').apply(parse_embedding).values)
    # Some rows might have varied length or empty; handle by padding/trimming
    # Find max dim
    dim = max(emb_train.shape[1], emb_test.shape[1])
    def pad_rows(mat, dim):
        n, d = mat.shape
        if d < dim:
            pad = np.zeros((n, dim - d))
            return np.hstack([mat, pad])
        return mat[:, :dim]
    emb_train = pad_rows(emb_train, dim)
    emb_test  = pad_rows(emb_test, dim)
    # PCA reduce to 8 comps
    pca = PCA(n_components=8, random_state=42)
    emb_all = np.vstack([emb_train, emb_test])
    pca.fit(emb_all)
    emb_train_p = pca.transform(emb_train)
    emb_test_p  = pca.transform(emb_test)
    for i in range(emb_train_p.shape[1]):
        meta[f'emb_pca_{i}'] = emb_train_p[:, i]
        test_meta[f'emb_pca_{i}'] = emb_test_p[:, i]
else:
    # no embeddings available
    pass

# ---------- 4. Feature engineering ----------
def feature_engineering(df):
    X = df.copy()
    # Numeric features (ensure no NaN)
    numeric_feats = ['price','life_cycle_length','num_stores','num_sizes','production']
    for c in numeric_feats:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
    # ratios and interactions
    if 'production' in X.columns and 'weekly_sales' in X.columns:
        # aggregated weekly_sales per id might be present — but if not, ignore
        X['sales_to_prod'] = X['weekly_sales'].fillna(0) / (X['production'].replace(0, np.nan).fillna(1))
    X['stores_x_sizes'] = X.get('num_stores',0) * X.get('num_sizes',0)
    # date/season encodings: use id_season, year, num_week_iso if available
    if 'year' in X.columns:
        X['year'] = X['year'].fillna(-1).astype(int)
    # simple encoding of categorical features counts
    cat_cols = ['aggregated_family','family','category','fabric','color_name','length_type',
                'silhouette_type','waist_type','sleeve_length_type','print_type','archetype','moment','ocassion']
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna('Unknown')
    # For simplicity return subset
    keep = ['ID','id_season','demand'] if 'demand' in X.columns else ['ID','id_season']
    keep += ['price','life_cycle_length','num_stores','num_sizes','production','stores_x_sizes','sales_to_prod']
    # add emb pca columns if present
    keep += [col for col in X.columns if col.startswith('emb_pca_')]
    # add categorical cols
    keep += [c for c in cat_cols if c in X.columns]
    # filter duplicates
    keep = [c for c in keep if c in X.columns]
    return X[keep]

train_fe = feature_engineering(meta)
test_fe  = feature_engineering(test_meta)

# ---------- 5. Prepare modeling matrices ----------
# Label encode categoricals for LightGBM (or use category param)
cat_cols = [c for c in train_fe.columns if train_fe[c].dtype=='object']
le_map = {}
for c in cat_cols:
    le = LabelEncoder()
    vals = list(train_fe[c].astype(str).unique()) + list(test_fe[c].astype(str).unique())
    le.fit(vals)
    train_fe[c] = le.transform(train_fe[c].astype(str))
    test_fe[c]  = le.transform(test_fe[c].astype(str))
    le_map[c] = le

# Define X,y
target_col = 'demand'
X = train_fe.drop(columns=[c for c in ['ID','id_season','demand'] if c in train_fe.columns])
y = train_fe[target_col].fillna(0).values
X_test = test_fe.drop(columns=[c for c in ['ID','id_season','demand'] if c in test_fe.columns])

# ---------- 6. Custom asymmetric squared loss for LightGBM ----------
# Penalize underestimation more strongly. Set alpha > 1 for underestimation weight.
alpha = 2.0  # e.g., underestimates counted 2x worse

def asymmetric_squared_obj(preds, dataset):
    # preds: raw predictions (not transformed)
    labels = dataset.get_label()
    resid = preds - labels
    # weight: alpha when preds < labels (underestimate), else 1
    w = np.where(resid < 0, alpha, 1.0)
    grad = 2.0 * w * resid
    hess = 2.0 * w
    return grad, hess

def asymmetric_eval(preds, dataset):
    labels = dataset.get_label()
    resid = preds - labels
    # asymmetric squared error
    w = np.where(resid < 0, alpha, 1.0)
    loss = np.mean(w * (resid**2))
    return 'asym_mse', loss, False

# ---------- 7. Time-aware CV and training ----------
# Use GroupKFold on id_season to avoid leakage (train on earlier seasons)
groups = train_fe['id_season'].values
# order seasons chronologically if possible; here GroupKFold as simple option
gkf = GroupKFold(n_splits=3)

oof = np.zeros(len(X))
preds_test = np.zeros(len(X_test))

features = X.columns.tolist()
lgb_params = {
    'boosting_type':'gbdt',
    'objective':'regression',
    'metric':'None',  # we'll use custom eval
    'learning_rate':0.05,
    'num_leaves':31,
    'min_data_in_leaf':50,
    'feature_fraction':0.8,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'verbosity': -1,
    'seed': 42
}

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dval],
        feval=asymmetric_eval,
        fobj=asymmetric_squared_obj,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    preds_test += model.predict(X_test, num_iteration=model.best_iteration) / gkf.n_splits

# ---------- 8. Simple post-processing ----------
# Predictions must be non-negative and possibly within [0,1] if dataset is scaled 0-1
preds_test = np.maximum(0, preds_test)
# If targets are scaled 0-1 in the dataset, test predictions are in same scale. If you want to
# add a family-level shrinkage: multiply by small factor if overpredicting historically — optional.

# ---------- 9. Create submission ----------
submission = sample_sub.copy()
# sample_sub likely contains id column, we must map preds_test to sample ids order
# Build dataframe with test ids and preds
test_ids = test_fe['ID'].values
pred_df = pd.DataFrame({'ID': test_ids, 'demand': preds_test})
# Ensure alignment with sample_sub (ID column could be named 'ID' or 'id')
id_col = submission.columns[0]
# merge
submission = submission.drop(columns=[c for c in submission.columns if c!='ID' and c!='ID'] , errors='ignore')
# Normalize merge by matching column name
if 'ID' in submission.columns and 'ID' not in submission.columns:
    submission = submission.merge(pred_df.rename(columns={'ID':'ID','demand':'demand'}), on='ID', how='left')
elif 'ID' in submission.columns:
    submission = submission.merge(pred_df, on='ID', how='left')
else:
    # fallback: use sample order
    submission['demand'] = preds_test[:len(submission)]
# Fill any missing with 0
submission['demand'] = submission['demand'].fillna(0)

submission[['ID','demand']].to_csv('submission.csv', index=False)
print('Wrote submission.csv')