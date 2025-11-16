import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
import lightgbm as lgb

# ---------- 1. Carregar dades ----------
train = pd.read_csv('train.csv', sep =';')
test  = pd.read_csv('test.csv', sep = ';')
sample_sub = pd.read_csv(sep = ',')

# ---------- 2. Sumar variable a predir ----------
# Com cada producte te varies files, aggregem weekly_demand per id
agg_target = train.groupby('ID', as_index=False)['weekly_demand'].sum().rename(columns={'weekly_demand':'demand'})
# Ajuntem demanda (agafem unique per id rows)
meta = train.drop_duplicates(subset=['ID'])  # si cada id te varies files, ens quedem nomes una
meta = meta.merge(agg_target, on='ID', how='left')

# Pel test, necessitem les files meta per a cada id en sample_submission (utilitzem files uniques de test.csv)
test_meta = test.drop_duplicates(subset=['ID']).copy()

# ---------- 3. Analitzar / extreure el vector image embedding ----------
# image_embedding guardat com "[0.12, 0.34, ...]"
def parse_embedding(s):
    try:
        return np.fromstring(s.strip("[]"), sep=',')
    except:
        return np.array([])

# Construir matriu pel PCA
if 'image_embedding' in meta.columns and meta['image_embedding'].notnull().any():
    emb_train = np.vstack(meta['image_embedding'].fillna('[]').apply(parse_embedding).values)
    emb_test = np.vstack(test_meta['image_embedding'].fillna('[]').apply(parse_embedding).values)
    # Algunes files podrien tenir longituds diferents o estar buides
    # Trobem dimenció màxima
    dim = max(emb_train.shape[1], emb_test.shape[1])
    def pad_rows(mat, dim):
        n, d = mat.shape
        if d < dim:
            pad = np.zeros((n, dim - d))
            return np.hstack([mat, pad])
        return mat[:, :dim]
    emb_train = pad_rows(emb_train, dim)
    emb_test  = pad_rows(emb_test, dim)
    # PCA redueix a 8 components
    pca = PCA(n_components=8, random_state=42)
    emb_all = np.vstack([emb_train, emb_test])
    pca.fit(emb_all)
    emb_train_p = pca.transform(emb_train)
    emb_test_p  = pca.transform(emb_test)
    for i in range(emb_train_p.shape[1]):
        meta[f'emb_pca_{i}'] = emb_train_p[:, i]
        test_meta[f'emb_pca_{i}'] = emb_test_p[:, i]
else:
    # no image_embeddings
    pass

# ---------- 4. Preparació de variables ----------
def feature_engineering(df):
    X = df.copy()
    # Característiques numèriques (assegurar que no hi ha NaN)
    numeric_feats = ['price','life_cycle_length','num_stores','num_sizes','production']
    for c in numeric_feats:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
    # Ràtios i interaccions
    if 'production' in X.columns and 'weekly_sales' in X.columns:
        # podria existir weekly_sales agregat per id si no, s'ignora
        X['sales_to_prod'] = X['weekly_sales'].fillna(0) / (X['production'].replace(0, np.nan).fillna(1))
    X['stores_x_sizes'] = X.get('num_stores',0) * X.get('num_sizes',0)
    # Utilitzar id_season, year, num_week_iso si existeixen
    if 'year' in X.columns:
        X['year'] = X['year'].fillna(-1).astype(int)
    # Codificació simple de característiques categòriques
    cat_cols = ['aggregated_family','family','category','fabric','color_name','length_type',
                'silhouette_type','waist_type','sleeve_length_type','print_type','archetype','moment','ocassion']
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna('Unknown')
    # Retornem un subconjunt de columnes
    keep = ['ID','id_season','demand'] if 'demand' in X.columns else ['ID','id_season']
    keep += ['price','life_cycle_length','num_stores','num_sizes','production','stores_x_sizes','sales_to_prod']
    # Afegir columnes PCA d’embeddings si existeixen
    keep += [col for col in X.columns if col.startswith('emb_pca_')]
    # Afegir columnes categòriques
    keep += [c for c in cat_cols if c in X.columns]
    # Filtrar duplicats
    keep = [c for c in keep if c in X.columns]
    return X[keep]

train_fe = feature_engineering(meta)
test_fe  = feature_engineering(test_meta)

# ---------- 5. Preparació de matrius per al model ----------
# Codificació d'etiquetes per a LightGBM (o ús directe de category param)
cat_cols = [c for c in train_fe.columns if train_fe[c].dtype=='object']
le_map = {}
for c in cat_cols:
    le = LabelEncoder()
    vals = list(train_fe[c].astype(str).unique()) + list(test_fe[c].astype(str).unique())
    le.fit(vals)
    train_fe[c] = le.transform(train_fe[c].astype(str))
    test_fe[c]  = le.transform(test_fe[c].astype(str))
    le_map[c] = le

# Definir X i y
target_col = 'demand'
X = train_fe.drop(columns=[c for c in ['ID','id_season','demand'] if c in train_fe.columns])
y = train_fe[target_col].fillna(0).values
X_test = test_fe.drop(columns=[c for c in ['ID','id_season','demand'] if c in test_fe.columns])

# ---------- 6. Pèrdua quadràtica asimètrica personalitzada per LightGBM ----------
# Penalitzar més fort les infraestimacions. Ajusta alpha > 1 per més pes.
alpha = 2.0  # per exemple, les infraestimacions es penalitzen el doble

def asymmetric_squared_obj(preds, dataset):
    # preds: prediccions en brut (no transformades)
    labels = dataset.get_label()
    resid = preds - labels
    # pes: alpha quan pred < label (infraestimació), si no 1
    w = np.where(resid < 0, alpha, 1.0)
    grad = 2.0 * w * resid
    hess = 2.0 * w
    return grad, hess

def asymmetric_eval(preds, dataset):
    labels = dataset.get_label()
    resid = preds - labels
    # error quadràtic asimètric
    w = np.where(resid < 0, alpha, 1.0)
    loss = np.mean(w * (resid**2))
    return 'asym_mse', loss, False

# ---------- 7. CV temporal i entrenament ----------
# Fer servir GroupKFold per id_season per evitar filtració temporal
groups = train_fe['id_season'].values # order seasons chronologically if possible; here GroupKFold as simple option 
gkf = GroupKFold(n_splits=3)

oof = np.zeros(len(X))
preds_test = np.zeros(len(X_test))

features = X.columns.tolist()
lgb_params = {
    'boosting_type':'gbdt',
    'objective':'regression',
    'metric':'None',  # farem servir l'avaluació personalitzada
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

# ---------- 8. Post-processament simple ----------
# Les prediccions han de ser no-negatives (i possiblement dins [0,1] si les dades estan escalades)
preds_test = np.maximum(0, preds_test)
# Si les etiquetes estan escalades a 0-1, les prediccions també. Es pot aplicar un ajustament
# addicional per família si cal.

# ---------- 9. Crear submissió ----------
submission = sample_sub.copy()
# sample_sub probablement conté la columna ID; cal alinear l’ordre
test_ids = test_fe['ID'].values
pred_df = pd.DataFrame({'ID': test_ids, 'demand': preds_test})

# Assegurar alineació segons la primera columna del sample
id_col = submission.columns[0]

# merge segur
submission = submission.drop(columns=[c for c in submission.columns if c!='ID' and c!='ID'] , errors='ignore')

# Normalitzar fusió segons el nom de la columna
if 'ID' in submission.columns and 'ID' not in submission.columns:
    submission = submission.merge(pred_df.rename(columns={'ID':'ID','demand':'demand'}), on='ID', how='left')
elif 'ID' in submission.columns:
    submission = submission.merge(pred_df, on='ID', how='left')
else:
    # alternativa: utilitzar l'ordre del sample
    submission['demand'] = preds_test[:len(submission)]

# Omplir buits amb 0
submission['demand'] = submission['demand'].fillna(0)

submission[['ID','demand']].to_csv('submission.csv', index=False)
print('S\'ha escrit submission.csv')
