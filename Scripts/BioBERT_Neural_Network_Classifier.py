import os
import time
import joblib
import numpy as np
import pandas as pd
from Scripts.functions_for_evaluation import AUCAUPkfold_from_file
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from tensorflow.keras import layers

import psutil
physical_cores = psutil.cpu_count(logical=False)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
data_dir = "Data"
model_dir = "Models"
metrics_dir = "Metrics"
embeddings_file = os.path.join(data_dir, "bio_bert_embeddings.tsv")
tuning_idx_file = os.path.join(model_dir, "tuning_data_indexes.csv")
training_folds_file = os.path.join(model_dir, "training_splits.pkl")
params_file = os.path.join(model_dir, "BioBERT_Neural_Network_Classifier_Params.pkl")
cv_metrics_file = os.path.join(metrics_dir, "BioBERT_Neural_Network_Classifier_CVmetrics.pkl")
metrics_prefix = os.path.join(metrics_dir, "BioBERT_Neural_Network_Classifier")
models_prefix= os.path.join(model_dir, "BioBERT_Neural_Network_Classifier")
timing_file = os.path.join(model_dir, "BioBERT_Neural_Network_Classifier_Timing.pkl")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# -------------------------------
# 1. Load Data
# -------------------------------
print(f"Loading data from {embeddings_file}...")
embeddings_df = pd.read_csv(embeddings_file, sep='\t')
embeddings_df['code'] = embeddings_df['code'].astype(str)
# Embedding features start from column 2
X = embeddings_df.iloc[:, 2:-1].apply(pd.to_numeric, errors='coerce').fillna(0).values
# Adjust labels (subtract 1)
y = (embeddings_df["nova_group"].astype(int) - 1).values
num_classes = len(np.unique(y))

print(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
print(f"Number of classes detected: {num_classes}")

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


# --------------------------------------------------------
# 3.1. Define a Small, Yet Flexible Neural Network Builder
# --------------------------------------------------------
def build_small_nn(input_dim, num_units=32, dropout_rate=0.0, learning_rate=0.001):
    """
    Build a small neural network with one hidden layer.
      - input_dim: Number of input features.
      - num_units: Number of units in the hidden layer.
      - dropout_rate: Dropout rate (0.0 means no dropout).
      - learning_rate: Learning rate for the Adam optimizer.
    """
    model = keras.Sequential()
    model.add(layers.InputLayer(shape=(input_dim,)))
    model.add(layers.Dense(num_units, activation='relu'))
    if dropout_rate > 0.0:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

input_dim = X_tune.shape[1]


# -------------------------------------------------
# 3.2. Wrap the Model with scikeras KerasClassifier
# -------------------------------------------------
nn_classifier = KerasClassifier(
    model=build_small_nn,
    input_dim=input_dim,
    epochs=5,         # Default epoch value; this will be tuned as well.
    batch_size=32,
    verbose=0         
)

# -----------------------------------------------------
# 3.3. Define Parameter Grid and Run RandomizedSearchCV
# -----------------------------------------------------
param_grid = {
    'model__num_units': [32, 64, 128],
    'model__dropout_rate': [0.0, 0.3, 0.5],
    'model__learning_rate': [0.001, 0.0005, 0.0001],
    'epochs': [5, 10, 15],
    'batch_size': [32, 64, 128]
}

n_iter = 50  

print("Starting expanded hyperparameter tuning for BERT NN...")
nn_random_search = RandomizedSearchCV(
    estimator=nn_classifier,
    param_distributions=param_grid,
    n_iter=n_iter,
    scoring='accuracy',
    cv=5,   
    n_jobs=1,     
    random_state=42,
    verbose=2
)

start_time = time.time()
with tqdm_joblib(tqdm(desc="NN Expanded Search", total=n_iter)):
    nn_random_search.fit(X_tune, y_tune)

param_search_time = time.time() - start_time

print("Best CV BERT NN Accuracy:", nn_random_search.best_score_)
print("Hyperparameter search time: {:.2f} seconds".format(param_search_time))
best_params = nn_random_search.best_params_
joblib.dump(best_params, params_file)

# ──────────────────────────────────────────────────────────────────────────────
# 4. 5‑fold CV on the remaining 80%
# ──────────────────────────────────────────────────────────────────────────────
start_cv = time.time()
auc, aup, models = AUCAUPkfold_from_file(
    X, y, 
    type='NN', 
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
