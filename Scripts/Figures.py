import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from seaborn import desaturate

# Define model paths
data_dir = "Data/"
model_dir = "Models/"
path_to_performance = "Metrics/"

models = {
    "BERT Random Forest": {"roc": path_to_performance + "BERT_Random_Forest_Classifier_no_class_balancing_ROC.pkl", "prc": path_to_performance + "BERT_Random_Forest_Classifier_no_class_balancing_PRC.pkl", "auc": path_to_performance + "BERT_Random_Forest_Classifier_no_class_balancing_AUC.pkl", "aup": path_to_performance + "BERT_Random_Forest_Classifier_no_class_balancing_AUP.pkl"},
    "BERT Neural Network": {"roc": path_to_performance + "BERT_Neural_Network_Classifier_ROC.pkl", "prc": path_to_performance + "BERT_Neural_Network_Classifier_PRC.pkl", "auc": path_to_performance + "BERT_Neural_Network_Classifier_AUC.pkl", "aup": path_to_performance + "BERT_Neural_Network_Classifier_AUP.pkl"},
    "BERT XGBoost": {"roc": path_to_performance + "BERT_XGBoost_Classifier_ROC.pkl", "prc": path_to_performance + "BERT_XGBoost_Classifier_PRC.pkl", "auc": path_to_performance + "BERT_XGBoost_Classifier_AUC.pkl", "aup": path_to_performance + "BERT_XGBoost_Classifier_AUP.pkl"},
    "BioBERT Random Forest": {"roc": path_to_performance + "BioBERT_Random_Forest_Classifier_no_class_balancing_ROC.pkl", "prc": path_to_performance + "BioBERT_Random_Forest_Classifier_no_class_balancing_PRC.pkl", "auc": path_to_performance + "BioBERT_Random_Forest_Classifier_no_class_balancing_AUC.pkl", "aup": path_to_performance + "BioBERT_Random_Forest_Classifier_no_class_balancing_AUP.pkl"},
    "BioBERT Neural Network": {"roc": path_to_performance + "BioBERT_Neural_Network_Classifier_ROC.pkl", "prc": path_to_performance + "BioBERT_Neural_Network_Classifier_PRC.pkl", "auc": path_to_performance + "BioBERT_Neural_Network_Classifier_AUC.pkl", "aup": path_to_performance + "BioBERT_Neural_Network_Classifier_AUP.pkl"},
    "BioBERT XGBoost": {"roc": path_to_performance + "BioBERT_XGBoost_Classifier_ROC.pkl", "prc": path_to_performance + "BioBERT_XGBoost_Classifier_PRC.pkl", "auc": path_to_performance + "BioBERT_XGBoost_Classifier_AUC.pkl", "aup": path_to_performance + "BioBERT_XGBoost_Classifier_AUP.pkl"},
    "Explanatory model - Ingredients": {"roc": path_to_performance + "Explanatory_model_num_of_ingredients_no_class_balancing_ROC.pkl", "prc": path_to_performance + "Explanatory_model_num_of_ingredients_no_class_balancing_PRC.pkl", "auc": path_to_performance + "Explanatory_model_num_of_ingredients_no_class_balancing_AUC.pkl", "aup": path_to_performance + "Explanatory_model_num_of_ingredients_no_class_balancing_AUP.pkl"},
    "Explanatory model - Additives": {"roc": path_to_performance + "Explanatory_model_num_of_additives_no_class_balancing_ROC.pkl", "prc": path_to_performance + "Explanatory_model_num_of_additives_no_class_balancing_PRC.pkl", "auc": path_to_performance + "Explanatory_model_num_of_additives_no_class_balancing_AUC.pkl", "aup": path_to_performance + "Explanatory_model_num_of_additives_no_class_balancing_AUP.pkl"},
    "FoodProx 11 nutrients": {"roc": path_to_performance + "FoodProX_model_11_nutrients_no_class_balancing_ROC.pkl", "prc": path_to_performance + "FoodProX_model_11_nutrients_no_class_balancing_PRC.pkl", "auc": path_to_performance + "FoodProX_model_11_nutrients_no_class_balancing_AUC.pkl", "aup": path_to_performance + "FoodProX_model_11_nutrients_no_class_balancing_AUP.pkl"},
    "FoodProX - 11 Nutrients + Additives": {"roc": path_to_performance + "FoodProX_model_11_nutrients_and_additives_no_class_balancing_ROC.pkl", "prc": path_to_performance + "FoodProX_model_11_nutrients_and_additives_no_class_balancing_PRC.pkl", "auc": path_to_performance + "FoodProX_model_11_nutrients_and_additives_no_class_balancing_AUC.pkl", "aup": path_to_performance + "FoodProX_model_11_nutrients_and_additives_no_class_balancing_AUP.pkl"}
}

# Define NOVA classes
nova_classes = [0, 1, 2, 3]

def compute_nova_class_prevalence(splits_path: str, off_file: str):
    # 1) load your CV-fold splits
    splits = joblib.load(splits_path)
    # 2) load OFF data and extract labels
    df_off = pd.read_csv(off_file, sep='\t')
    # zero-index the nova_group
    y = (df_off['nova_group'].astype(int) - 1).values
    N = len(y)
    # 3) gather all test indices (filtered to the valid range)
    all_test_idx = []
    for fold in splits:
        te = fold['test']
        all_test_idx.extend(i for i in te if i < N)
    # 4) compute per-class prevalence in the pooled test set
    all_y_test = y[all_test_idx]
    num_classes = len(np.unique(y))
    prevalence = [(all_y_test == c).mean() for c in range(num_classes)]
    return prevalence

embeddings_file=data_dir+"Filtered_OFF_with_sentences.csv"
splits_path=model_dir+"training_splits.pkl"

prevalence = compute_nova_class_prevalence(splits_path, embeddings_file)

print("NOVA Class Prevalence:", prevalence)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Plot Raw ROC / PRC Curves
# ──────────────────────────────────────────────────────────────────────────────
def plot_raw_curves(metric_type, save_path, xlabel, ylabel):
    fig, axes = plt.subplots(2,2,figsize=(12,10))
    axes = axes.flatten()
    cmap = plt.cm.get_cmap("tab10", len(models))
    for ci, cls in enumerate(nova_classes):
        ax = axes[ci]
        for mi, (mname, paths) in enumerate(models.items()):
            data = joblib.load(paths[metric_type])
            for f in range(len(data)):
                x, y = data[f][ci]   
                ax.plot(x, y, color=cmap(mi), alpha=0.3, lw=1)
            ax.plot([], [], color=cmap(mi), lw=2, label=mname)
        ax.set_title(f"NOVA {cls+1}", fontsize=14)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved raw {metric_type.upper()} curves to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Plot Bar‐Charts of AUC / AUP Scores (mean±std)
# ──────────────────────────────────────────────────────────────────────────────
def plot_auc_aup(metric_type, save_path, ylabel):
    fig, ax = plt.subplots(figsize=(12,6))
    bar_w = 0.08
    num_classes=4
    spacing = 0.4
    M = len(models)
    palette = sns.color_palette("pastel", 10)
    cmap = ListedColormap(palette)
    x0 = np.arange(num_classes)*(M*bar_w + spacing)
    for mi, (mname, paths) in enumerate(models.items()):
        scores = joblib.load(paths[metric_type])  
        mean_scores = scores.mean(axis=0)
        std_scores = scores.std(axis=0)
        xs = x0 + mi*bar_w
        ax.bar(xs, mean_scores, bar_w, yerr=std_scores, label=mname, color=cmap(mi))
        for ci, sc in enumerate(mean_scores):
            if metric_type == "aup":
                if sc < 0.2:
                    ypos = 0.05       
                    va = "bottom"
                else:
                    ypos = sc / 2    
                    va = "center"
            else:
                ypos = 0.5 + abs(sc-0.5)/2
            ax.text(xs[ci], ypos, f"{sc:.3f}", ha="center", va="center", rotation=90, fontsize=8)
    ax.set_xticks(x0 + M*bar_w/2)
    ax.set_xticklabels([f"NOVA {c+1}" for c in nova_classes])
    ax.set_ylabel(ylabel)
    if metric_type == "auc":
        ax.axhline(0.5, ls="--", color="gray", label="Random (AUC)")
    else:
        for ci, base in enumerate(prevalence):
            start = x0[ci] - bar_w/2
            end = x0[ci] + M*bar_w
            ax.hlines(base, start, end, ls="--", color="gray")
        ax.plot([], [], ls="--", color="gray", label="Random (AUP)")
    ax.legend(fontsize=8, bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {metric_type.upper()} bar chart to {save_path}")




def plot_interp_curves(save_path,xlabel,ylabel,metric_type):
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    axes=axes.flatten()
    cmap=plt.cm.get_cmap("tab10",len(models))
    grid=np.linspace(0,1,500)
    for ci,cls in enumerate(nova_classes):
        ax=axes[ci]
        for mi,(mname,paths) in enumerate(models.items()):
            raw=joblib.load(paths[metric_type])
            interp_curves=[]
            for fold in raw:
                x,y=fold[ci]
                order=np.argsort(x)
                xi,yi=x[order],y[order]
                interp_curves.append(np.interp(grid,xi,yi))
            arr=np.vstack(interp_curves)
            mean_curve=arr.mean(axis=0)
            std_curve=arr.std(axis=0)
            ax.plot(grid,mean_curve,color=cmap(mi),lw=2,label=mname)
            ax.fill_between(grid,mean_curve-std_curve,mean_curve+std_curve,color=cmap(mi),alpha=0.2)
        ax.set_title(f"NOVA {cls+1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8,loc="best")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved interpolated {metric_type.upper()} curves to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Generate & Save All Plots
# ──────────────────────────────────────────────────────────────────────────────
plot_raw_curves("roc", path_to_performance + "NOVA_raw_ROC_Curves.pdf", "False Positive Rate", "True Positive Rate")
plot_raw_curves("prc", path_to_performance + "NOVA_raw_PRC_Curves.pdf", "Recall", "Precision")
plot_interp_curves(path_to_performance+"NOVA_interp_ROC_Curves.pdf", "False Positive Rate", "True Positive Rate", metric_type="roc")
plot_interp_curves(path_to_performance + "NOVA_interp_PRC_Curves.pdf","Recall","Precision",metric_type="prc")
plot_auc_aup("auc", path_to_performance + "NOVA_AUC_Scores_no_class_balancing.pdf", "AUC Score")
plot_auc_aup("aup", path_to_performance + "NOVA_AUP_Scores_no_class_balancing.pdf", "AUP Score")



import pandas as pd

timings_csv = "/home/g.ispirova/GDB/Metrics/timings_summary.csv"
df = pd.read_csv(timings_csv)

print("Columns in timings_summary.csv:", df.columns.tolist())

# Only pick up columns that are numeric (ints or floats), not the *_readable strings
numeric_time_cols = [
    c for c, dt in df.dtypes.items()
    if c != "model" and dt.kind in ("i","u","f")
]

print("Detected numeric time columns:", numeric_time_cols)

def format_duration(sec):
    if pd.isna(sec):
        return ""
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"

for col in numeric_time_cols:
    df[f"{col}_readable"] = df[col].apply(format_duration)

# Now print the human‐readable columns
display_cols = ["model"] + [f"{col}_readable" for col in numeric_time_cols]
print("\nTiming Summary (human‐readable):")
print(df[display_cols].to_string(index=False))

# And overwrite your CSV so it includes the new *_readable columns
df.to_csv(timings_csv, index=False)
print(f"\nUpdated timing summary saved to {timings_csv}")
