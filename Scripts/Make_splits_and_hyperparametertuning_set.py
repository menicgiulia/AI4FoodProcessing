import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# ── Paths ─────────────────────────────────────────────────────────────
data_dir = "Data"
model_dir = "Models"
embeddings_file = os.path.join(data_dir, "bert_embeddings.tsv")
tuning_idx_file = os.path.join(model_dir, "tuning_data_indexes.csv")
training_splits_file= os.path.join(model_dir, "training_splits.pkl")

os.makedirs(model_dir, exist_ok=True)

# ── 1. Load embeddings + labels ─────────────────────────────────────────
emb = pd.read_csv(embeddings_file, sep="\t")
emb['code'] = emb['code'].astype(str)
X = emb.iloc[:, 2:-1].apply(pd.to_numeric, errors='coerce').fillna(0).values
y = (emb['nova_group'].astype(int) - 1).values
print(f"Loaded {X.shape[0]} samples, {X.shape[1]} dims, {len(np.unique(y))} classes")

# ── 2. Carve off a stratified 20% tuning set ─────────────────────────────
all_idx = np.arange(len(y))
tune_idx, rest_idx = train_test_split(
    all_idx,
    test_size=0.80,
    stratify=y,
    random_state=42
)

# Save for reproducibility
pd.DataFrame({'index': tune_idx}).to_csv(tuning_idx_file, index=False)
print(f"Saved 20% tuning indexes → {tuning_idx_file}")

# ── 3. Build 5 stratified folds on the remaining 80% ────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = []
# skf.split wants X_rest, y_rest
X_rest = X[rest_idx]
y_rest = y[rest_idx]
for train_loc, test_loc in skf.split(X_rest, y_rest):
    splits.append({
        'train': rest_idx[train_loc].tolist(),
        'test':  rest_idx[test_loc].tolist()
    })

# ── 4. Save those 5 splits as a pickled list of dicts ───────────────────
joblib.dump(splits, training_splits_file)
print(f"Saved 5 train/test splits → {training_splits_file}")
