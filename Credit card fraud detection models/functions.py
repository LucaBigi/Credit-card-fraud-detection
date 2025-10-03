import numpy as np
import pandas as pd
import torch
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix as skl_confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def stratified_split(csv_path, seed, test_size=0.2):
    df = pd.read_csv(csv_path)
    train, test= train_test_split(df,
        test_size=test_size,
        stratify=df["Class"],
        random_state=seed)
    return train, test

def print_confusion_matrix_basic(y_test, Y_test):
    cm = skl_confusion_matrix(Y_test, y_test)
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    print("\nConfusion Matrix:")
    print(f"True 0    {GREEN}[{cm[0, 0]}]{RESET}  {YELLOW}[{cm[0, 1]}]{RESET}")
    print(f"True 1      {RED}[{cm[1, 0]}]{RESET}  {GREEN}[{cm[1, 1]}]{RESET}")
    print("       Pred: 0      1")
    print("\n")

def get_balanced_indices(y, indices, seed):
    rng = np.random.default_rng(seed)
    y_ = y[indices]
    pos_idx = indices[y_ == 1]
    neg_idx = indices[y_ == 0]
    # Numero di campioni da estrarre (il minimo tra le due classi)
    n_samples = min(len(pos_idx), len(neg_idx))
    pos_sampled = rng.choice(pos_idx, size=n_samples, replace=False)
    neg_sampled = rng.choice(neg_idx, size=n_samples, replace=False)
    balanced_indices = np.concatenate([pos_sampled, neg_sampled])
    return balanced_indices

def get_balanced_indices_bin(bin_feat, y, indices, seed):
    rng = np.random.default_rng(seed)
    y_ = y[indices]
    pos_idx = indices[y_ == 1]
    neg_idx = indices[y_ == 0]
    n_pos = len(pos_idx)
    # Suddivisione in n_pos bin
    bins = np.linspace(bin_feat.min(), bin_feat.max(), n_pos + 1)
    neg_bins = np.digitize(bin_feat[neg_idx], bins) - 1
    # Seleziona un negativo per bin (quando disponibile)
    selected_neg = []
    for b in range(n_pos):
        candidates = neg_idx[neg_bins == b]
        if len(candidates) > 0:
            chosen = rng.choice(candidates)
            selected_neg.append(chosen)
    # Se non bastano, completa pescando random tra i rimanenti
    if len(selected_neg) < n_pos:
        remaining = np.setdiff1d(neg_idx, selected_neg, assume_unique=True)
        extra = rng.choice(remaining, size=n_pos - len(selected_neg), replace=False)
        selected_neg.extend(extra)
    return np.concatenate([pos_idx, np.array(selected_neg)])

def margin_split(model, X, scaler=None):
    if scaler is not None:
        X = scaler.transform(X)
    abs_decision = np.abs(model.decision_function(X))
    mask = abs_decision < 1
    inside  = np.flatnonzero(mask)    # punti dentro il margine
    outside = np.flatnonzero(~mask)   # punti fuori dal margine
    return inside, outside

def evaluate_model_performance(y_true, y_pred, y_prob, set_name=""):
    print(f"\nMatrice di confusione sul {set_name.upper()} set:")
    print_confusion_matrix_basic(y_pred, y_true)

    # Precision e Recall
    precision_ = precision_score(y_true, y_pred)
    recall_ = recall_score(y_true, y_pred)
    accuracy_ = accuracy_score(y_true, y_pred)
    print(f"Precision {set_name.lower()} set: {precision_:.4f}")
    print(f"Recall {set_name.lower()} set:    {recall_:.4f}")
    print(f"Accuracy {set_name.lower()} set:  {accuracy_:.4f}")

    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    print(f"AP ({set_name} set): {ap:.4f}")
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve {set_name} set')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"AUC ({set_name} set): {roc_auc:.4f}")
    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) {set_name} set")
    plt.legend(loc="lower right")
    plt.show()