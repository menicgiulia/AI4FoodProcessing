import os
import time
import joblib
import numpy as np
import pandas as pd
from Scripts.functions_for_evaluation import AUCAUPkfold_from_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import re

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
data_dir = "Data"
model_dir = "Models"
metrics_dir = "Metrics"
off_file            = os.path.join(data_dir, "Filtered_OFF_with_sentences.csv")
tuning_idx_file     = os.path.join(model_dir, "tuning_data_indexes.csv")
training_folds_file = os.path.join(model_dir, "training_splits.pkl")
params_file   = os.path.join(model_dir, "Explanatory_model_num_of_ingredients_Params.pkl")
cv_metrics_file     = os.path.join(metrics_dir, "Explanatory_model_num_of_ingredients_CVmetrics.pkl")
metrics_prefix  = os.path.join(metrics_dir, "Explanatory_model_num_of_ingredients")
models_prefix= os.path.join(model_dir, "Explanatory_model_num_of_ingredients")
timing_file         = os.path.join(model_dir, "Explanatory_model_num_of_ingredients_Timing.pkl")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# -------------------------------
# 1. Load & Prepare OFF Dataset
# -------------------------------
print(f"Loading OFF data from {off_file}...")
df_off = pd.read_csv(off_file, sep='\t')
# Drop rows missing 'ingredients_text' or 'nova_group'
df_off = df_off.dropna(subset=["ingredients_text", "nova_group"]).reset_index(drop=True)

def clean_ingredients(text):
    no_paren = re.sub(r"\([^)]*\)", "", text)
    no_brack = re.sub(r"\[[^]]*\]", "", no_paren)
    parts = [tok.strip() for tok in no_brack.split(',') if tok.strip()]
    return len(parts)

# compute number of ingredients
df_off['num_ingredients'] = df_off['ingredients_text'].astype(str).apply(clean_ingredients)

# Prepare feature matrix X and label y
X = df_off[['num_ingredients']].values
y = (df_off['nova_group'].astype(int) - 1).values
num_classes = len(np.unique(y))

print(f"Prepared OFF data: X shape {X.shape}, y shape {y.shape}, classes: {num_classes}")
print(f"Detected {num_classes} NOVA classes.")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Load tuning & fold indexes
# ──────────────────────────────────────────────────────────────────────────────
tune_idx = pd.read_csv(tuning_idx_file)['index'].values
df_folds = joblib.load(training_folds_file)
splits = [
    ( np.array(d['train'], dtype=int),
      np.array(d['test'],  dtype=int) )
    for d in df_folds
]
# ──────────────────────────────────────────────────────────────────────────────
# 3. Hyperparameter tuning on the 20%
# ──────────────────────────────────────────────────────────────────────────────
X_tune, y_tune = X[tune_idx], y[tune_idx]


n_estimators = [int(x) for x in np.linspace(200, 2000, num=10)]
max_features = ['log2','sqrt']
max_depth    = [int(x) for x in np.linspace(100,500,num=11)] + [None]

grid = {
    'n_estimators':   n_estimators,
    'max_features':   max_features,
    'max_depth':      max_depth
}

""" search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=grid,
    n_iter=50,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=2
) """

from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=grid,      
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=2,
    refit=True                    
)

start_tune = time.time()
with tqdm_joblib(tqdm(desc="RF tuning on 20%", total=50)):
    search.fit(X_tune, y_tune)


param_search_time = time.time() - start_tune

best_params = search.best_params_
joblib.dump(best_params, params_file)

# ──────────────────────────────────────────────────────────────────────────────
# 4. 5‑fold CV on the remaining 80%
# ──────────────────────────────────────────────────────────────────────────────
start_cv = time.time()
auc, aup, models = AUCAUPkfold_from_file(
    X, y,
    type="RF",
    params_file=params_file,
    splits=df_folds,
    models_prefix=models_prefix,
    metrics_prefix=metrics_prefix,
    verbose=True
)

cv_time = time.time() - start_cv

cv_summary = {
    'auc_mean':  (auc.mean(), auc.std()),
    'auprc_mean':(aup.mean(), aup.std())
}


joblib.dump(cv_summary, cv_metrics_file)

timing = {
    'param_search_time': param_search_time,
    'cv_time':           cv_time
}

joblib.dump(timing, timing_file)
