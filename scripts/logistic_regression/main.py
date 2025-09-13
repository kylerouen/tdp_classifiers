import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    matthews_corrcoef,
    log_loss,
    roc_auc_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

def main():
    # Load and clean data
    df = pd.read_csv("final.csv")
    df = df.drop(["optn_LE", "cavRef_LE", "optN_n_LGFE", "smiles"], axis=1)
    df = df.dropna()
    df["Risk"] = df["Risk"].replace({
        "PR": "PR+CR+SR", "CR": "PR+CR+SR", "SR": "PR+CR+SR"
    })

    # Filter for binary classification
    df = df[df.Risk.isin(["KR", "NR"])]

    # Prepare feature and target data
    X_df = df.drop(["Ligand", "Risk", "TdP"], axis=1)
    X_cols = X_df.columns
    X = X_df.values
    Y = df["TdP"].astype(int).values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- PCA Plot ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("Set1", n_colors=len(np.unique(Y)))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=Y, palette=palette, s=100)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA of Features')
    plt.legend(title='TdP')
    plt.tight_layout()
    plt.savefig("pca_plot.png", dpi=300)
    plt.show()

    # --- 5-Fold Cross-Validation ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    recalls = []
    specificities = []
    mccs = []
    aics = []
    f1s = []
    aucs = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, Y), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        clf = LogisticRegression(class_weight=None, max_iter=1000)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        pred_proba = clf.predict_proba(X_test)

        cm = confusion_matrix(y_test, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp)
        else:
            specificity = float("nan")

        acc = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions, average="macro")
        mcc = matthews_corrcoef(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, pred_proba[:, 1])

        # Approximate AIC
        logloss = log_loss(y_test, pred_proba)
        k = X_train.shape[1] + 1  # number of features + intercept
        aic = 2 * k + 2 * logloss * len(y_test)

        accuracies.append(acc)
        recalls.append(recall)
        specificities.append(specificity)
        mccs.append(mcc)
        f1s.append(f1)
        aucs.append(auc)
        aics.append(aic)

        print(f"Fold {fold}:")
        print(f"  Accuracy     = {acc:.3f}")
        print(f"  Specificity  = {specificity:.3f}")
        print(f"  Recall       = {recall:.3f}")
        print(f"  F1 Score     = {f1:.3f}")
        print(f"  ROC AUC      = {auc:.3f}")
        print(f"  MCC          = {mcc:.3f}")
        print(f"  AIC (approx) = {aic:.1f}")

    # --- Summary Stats ---
    print("\n--- Cross-Validation Summary ---")
    print(f"Mean Accuracy:     {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"Mean Specificity:  {np.nanmean(specificities):.3f} ± {np.nanstd(specificities):.3f}")
    print(f"Mean Recall:       {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    print(f"Mean F1 Score:     {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"Mean ROC AUC:      {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"Mean MCC:          {np.mean(mccs):.3f} ± {np.std(mccs):.3f}")
    print(f"Mean AIC (approx): {np.mean(aics):.1f} ± {np.std(aics):.1f}")

    # --- Coefficients from last fold ---
    print("\nLogistic Regression Coefficients (from last fold):")
    for predictor, coef in zip(X_cols, clf.coef_[0]):
        print(f"{predictor}: {coef:.4f}")

if __name__ == '__main__':
    main()
