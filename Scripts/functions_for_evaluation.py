import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import operator
import joblib
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import h5py

num_classes=4

def multiclass_roc_auc_score(y_test, y_probs, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_probs, average=average)

def multiclass_average_precision_score(y_test, y_probs, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return average_precision_score(y_test, y_probs, average=average)

def multiclass_roc_curve(y_test, y_probs):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    fpr = dict()
    tpr = dict()
    for i in range(y_probs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_probs[:, i])
    return (fpr, tpr)

def multiclass_average_precision_curve(y_test, y_probs):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    precision = dict()
    recall = dict()
    for i in range(y_probs.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_probs[:, i])
    return (precision, recall)

def build_small_nn(input_dim, num_units=32, dropout_rate=0.0, learning_rate=0.001):
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



def AUCAUPkfold_from_file(X, y, model_type, params_file, splits, models_prefix, metrics_prefix, verbose=True):
    best_params = joblib.load(params_file)
    classes = np.unique(y)
    n_classes = len(classes)
    n_folds = len(splits)
    perf_auc = np.zeros((n_folds, n_classes))
    perf_aup = np.zeros((n_folds, n_classes))
    rocs = []
    prcs = []
    models = []
    for i, fold in enumerate(splits):
        tr, te = fold['train'], fold['test']
        if model_type == 'RF':
            clf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
            clf.fit(X[tr], y[tr])
            y_proba = clf.predict_proba(X[te])
            models.append(clf)
        elif model_type == 'XGB':
            clf = XGBClassifier(random_state=42,
                                use_label_encoder=False,
                                eval_metric='logloss',
                                tree_method='hist',
                                verbosity=0,
                                **best_params)
            clf.fit(X[tr], y[tr])
            y_proba = clf.predict_proba(X[te])
            models.append(clf)
        elif model_type == 'NN':
            model = build_small_nn(
                input_dim=X.shape[1],
                num_units=best_params.get('model__num_units', 32),
                dropout_rate=best_params.get('model__dropout_rate', 0.0),
                learning_rate=best_params.get('model__learning_rate', 1e-3)
            )
            model.fit(
                X[tr], y[tr],
                epochs=best_params.get('epochs', 5),
                batch_size=best_params.get('batch_size', 32),
                verbose=0
            )
            y_proba = model.predict(X[te])
            models.append(model)
        else:
            raise ValueError(f"Unknown model_type {model_type!r}")
        y_true = y[te]
        fold_rocs = []
        fold_prcs = []
        for ci, c in enumerate(classes):
            y_bin = (y_true == c).astype(int)
            scores = y_proba[:, ci]
            perf_auc[i, ci] = roc_auc_score(y_bin, scores)
            perf_aup[i, ci] = average_precision_score(y_bin, scores)
            fpr, tpr, _ = roc_curve(y_bin, scores)
            fold_rocs.append((fpr, tpr))
            prec, rec, _ = precision_recall_curve(y_bin, scores)
            fold_prcs.append((prec, rec))
        rocs.append(fold_rocs)
        prcs.append(fold_prcs)
    if verbose:
        print("AUC mean:", perf_auc.mean(axis=0))
        print("AUC std: ", perf_auc.std(axis=0))
        print("AUP mean:", perf_aup.mean(axis=0))
        print("AUP std: ", perf_aup.std(axis=0))
    joblib.dump(perf_auc, metrics_prefix + "_AUC.pkl")
    joblib.dump(perf_aup, metrics_prefix + "_AUP.pkl")
    joblib.dump(rocs,     metrics_prefix + "_ROC.pkl")
    joblib.dump(prcs,     metrics_prefix + "_PRC.pkl")
    if model_type in ('RF', 'XGB'):
        joblib.dump(models, models_prefix + "_models.pkl")
    else:  
        with h5py.File(models_prefix + "_models.h5", "w") as hf:
            for idx, m in enumerate(models):
                grp = hf.create_group(f"fold_{idx}")
                grp.attrs["config"] = m.to_json()
                wgrp = grp.create_group("weights")
                for layer in m.layers:
                    lg = wgrp.create_group(layer.name)
                    for weight_tensor, array in zip(layer.weights, layer.get_weights()):
                        lg.create_dataset(weight_tensor.name, data=array)
    return perf_auc, perf_aup, models

