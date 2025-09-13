import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score
)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'DejaVu Sans']
from shap import force_plot

def main():
    # Load and preprocess
    df = pd.read_csv("final.csv")
    df = df.drop(["optn_LE", "cavRef_LE", "optN_n_LGFE", "smiles"], axis=1).dropna()
    df["Risk"] = df["Risk"].replace({
        "PR": "PR+CR+SR", "CR": "PR+CR+SR", "SR": "PR+CR+SR"
    })
    df = df[df["Risk"].isin(["KR", "NR"])].reset_index(drop=True)

    # Features and labels
    X_df = df.drop(["Ligand", "Risk", "TdP"], axis=1)
    X_cols = X_df.columns
    X = X_df.values
    Y = df["TdP"].astype(int).values

    # Train/test split
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, Y, df.index, test_size=0.2, random_state=42, stratify=Y
    )


    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    predictions = clf.predict(X_test)
    pred_probs = clf.predict_proba(X_test)[:, 1]

    # Metrics
    cm = confusion_matrix(y_test, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
    else:
        specificity = float("nan")

    print(f"\n=== Performance ===")
    print(f"Accuracy:     {accuracy_score(y_test, predictions):.3f}")
    print(f"Specificity:  {specificity:.3f}")
    print(f"Recall:       {recall_score(y_test, predictions):.3f}")
    print(f"F1 Score:     {f1_score(y_test, predictions):.3f}")
    print(f"MCC:          {matthews_corrcoef(y_test, predictions):.3f}")
    print(f"ROC AUC:      {roc_auc_score(y_test, pred_probs):.3f}")

    # Get false negatives: true label = 1, predicted = 0
    false_negatives_idx = np.where((y_test == 1) & (predictions == 0))[0]

    # Get original dataframe indices of those false negatives
    fn_original_idx = [test_idx[i] for i in false_negatives_idx]

    # Get names and predicted probs
    false_negatives_names = df.loc[fn_original_idx, "Ligand"].values
    false_negatives_probs = pred_probs[false_negatives_idx]

    # Print
    print("\nFalse Negatives (True=1, Predicted=0):")
    for name, prob in zip(false_negatives_names, false_negatives_probs):
        print(f"{name}: {prob:.3f}")


    # Assuming clf is your RandomForestClassifier and X_test, X_cols are ready

    explainer = shap.TreeExplainer(clf)
    shap_expl = explainer(X_test)  # Note: call explainer as a function

    # shap_expl.values shape is (samples, features, classes)
    print("SHAP values shape:", shap_expl.values.shape)

    X_test_df = pd.DataFrame(X_test, columns=X_cols)

    # Plot for class 0
    shap.summary_plot(shap_expl.values[:, :, 0], X_test_df, max_display=15, show=False)
    plt.title("SHAP Summary Plot for Class 0 (Negative)")
    #plt.show()

    # Plot for class 1
    shap.summary_plot(shap_expl.values[:, :, 1], X_test_df,  show=False)
    plt.title("SHAP Summary Plot for Class 1 (Positive)")
    #plt.show()

    ## Force plot for a selected drug (e.g., "Amiodarone")
    # Force plot for a selected drug (e.g., "Amiodarone")
    drug_name = "Amiodarone"
    match = df[df["Ligand"].str.contains(drug_name, case=False)]

    if not match.empty:
        drug_index = match.index[0]
        drug_X = X[drug_index:drug_index+1]  # Keep it 2D

        # Compute SHAP Explanation for the single sample
        shap_expl_drug = explainer(drug_X)

        # Choose class 1 explanation (e.g., TdP = 1)
        drug_shap = shap_expl_drug[..., 1]

        # Plot force plot with matplotlib backend
        shap.plots.force(drug_shap[0], matplotlib=True)
        plt.title(f"SHAP Force Plot for {drug_name}")
        plt.show()

    else:
        print(f"\nDrug '{drug_name}' not found in the dataset.")


if __name__ == "__main__":
    main()
