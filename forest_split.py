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
# Change font size globally
plt.rcParams.update({'font.size': 12})  # Set your desired font size
from shap import force_plot

def main():
    # Load and preprocess
    df = pd.read_csv("final.csv")
    df = df.drop(["optn_LE", "cavRef_LE", "optN_n_LGFE", "smiles"], axis=1).dropna()
    df["Risk"] = df["Risk"].replace({
        "PR": "PR+CR+SR", "CR": "PR+CR+SR", "SR": "PR+CR+SR"
    })
    df = df[df["Risk"].isin(["KR", "NR"])].reset_index(drop=True)

    # Find nonzero MAMC values
    nonzero_mamc_df = df[df["MAMC"] != 0]

    # Print Ligand names and their MAMC values
    print("=== Drugs with nonzero MAMC ===")
    for name, mamc in zip(nonzero_mamc_df["Ligand"], nonzero_mamc_df["MAMC"]):
        print(f"{name}: {mamc}")

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
    #clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=74,
        max_depth=24,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42  # optional, for reproducibility
    )
    # Fit the model
    clf.fit(X_train, y_train)

    # Get predicted probabilities for class 1
    pred_probs = clf.predict_proba(X_test)[:, 1]

    # Apply custom threshold
    custom_threshold = 0.4214
    predictions = (pred_probs > custom_threshold).astype(int)

    test_drug_names = df.loc[test_idx, "Ligand"].values
    print("\n=== Test Set Predictions ===")
    for idx, name, true_label, pred_label in zip(test_idx, test_drug_names, y_test, predictions):
        print(f"{name}: True={true_label}, Predicted={pred_label}")

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
    print(confusion_matrix(y_test, predictions))

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

    explainer = shap.TreeExplainer(clf, model_output="raw")
    shap_expl = explainer(X_test)  # Note: call explainer as a function

    # shap_expl.values shape is (samples, features, classes)
    print("SHAP values shape:", shap_expl.values.shape)

    X_test_df = pd.DataFrame(X_test, columns=X_cols)

    # Plot for class 0
    shap.summary_plot(shap_expl.values[:, :, 0], X_test_df, max_display=15, show=False)
    plt.title("SHAP Summary Plot for Class 0 (Negative)")
    plt.show()

    # Plot for class 1
    shap.summary_plot(shap_expl.values[:, :, 1], X_test_df, max_display=8,  show=False)
    plt.title("SHAP Summary Plot for Class 1 (Positive)")
    plt.show()

    # Compute mean absolute SHAP values across class 1
    mean_abs_shap = np.abs(shap_expl.values[:, :, 1]).mean(axis=0)

    # Get indices of top 7 features
    top_7_idx = np.argsort(mean_abs_shap)[::-1][:7]

    # Index of the specific feature you want to include
    target_feature = "Nav1.5_MEOO"
    target_idx = list(X_cols).index(target_feature)

    # Combine and deduplicate indices
    selected_indices = list(set(top_7_idx.tolist() + [target_idx]))

    # Get selected feature names and SHAP values
    selected_feature_names = [X_cols[i] for i in selected_indices]
    X_selected = X_test_df[selected_feature_names]
    shap_selected_values = shap_expl.values[:, selected_indices, 1]

    # Plot
    shap.summary_plot(shap_selected_values, X_selected, show=False)
    plt.title("Top 7 SHAP Features + MAMC")
    plt.show()

    ## Force plot for a selected drug (e.g., "Amiodarone")
    # Force plot for a selected drug (e.g., "Amiodarone")
    drug_name = "donepezil"
    match = df[df["Ligand"].str.contains(drug_name, case=False)]

    if not match.empty:
        drug_index = match.index[0]
        print("Drug selected for SHAP force plot:", df.loc[drug_index, "Ligand"])
        drug_X = X[drug_index:drug_index+1]  # Keep it 2D

        # Compute SHAP Explanation for the single sample
        shap_expl_drug = explainer(drug_X)

        # Choose class 1 explanation (e.g., TdP = 1)
        drug_shap = shap_expl_drug[..., 1]

        rounded_drug_X = np.round(drug_X, 3)
        blanked_features = ["–" for _ in range(len(X_cols))]

        # Plot force plot with matplotlib backend
        #shap.initjs()


        # Get SHAP values and feature values
        shap_vals = drug_shap.values[0]
        feat_vals = drug_X[0]

        # Identify top 5 contributors by absolute SHAP value
        top_k = 6
        top_indices = np.argsort(np.abs(shap_vals))[::-1][:top_k]

        # Create masked versions of feature names and values
        masked_names = [X_cols[i] if i in top_indices else "" for i in range(len(X_cols))]
        masked_vals = [feat_vals[i] if i in top_indices else "" for i in range(len(X_cols))]

        shap.plots.force(
            drug_shap[0],
            matplotlib=True,
            feature_names=masked_names,
            features=masked_vals,
            show=False
        )
        print("Base value:", explainer.expected_value[1])  # For class 1
        print("Model prediction (f(x)):", clf.predict_proba(X_test)[0, 1])
        print("Sum of SHAP values:", shap_expl.values[0, :, 1].sum())
        # Set custom x-axis range (e.g., SHAP value from -1 to +1)
        #plt.xlim(0.0, 0.6)
        # Explicitly set font size for all text objects in the plot
        for text_obj in plt.gca().texts:
            text_obj.set_fontsize(15)  # Set your desired font size here
        #plt.title("SHAP Force Plot (Top 5 Features)")
        plt.tight_layout()
        # ax = plt.gca()
        # for child in ax.get_children():
        #     if isinstance(child, plt.Line2D):
        #         if "--" in child.get_linestyle():  # Dashed line
        #             child.set_visible(False)
        #
        # # You can also remove all vertical lines (if above fails)
        # for line in ax.lines:
        #     if line.get_xdata()[0] == line.get_xdata()[1]:  # vertical line
        #         line.set_visible(False)
        #
        # # Optionally remove text annotations (if base value shows as text)
        # for txt in ax.texts:
        #     if "base value" in txt.get_text().lower() or txt.get_text().strip().startswith("Base"):
        #         txt.set_visible(False)
        plt.show()
        # Save or display
        #shap.save_html("force_plot_top5.html", shap_force)

        # drug_force_plot = shap.plots.force(drug_shap[0], feature_names=X_cols, show=False, features=blanked_features)
        #
        # plt.title(f"SHAP Force Plot for {drug_name}")
        # shap.save_html("ciproflox_force_plot.html", drug_force_plot)
        #plt.show()

    else:
        print(f"\nDrug '{drug_name}' not found in the dataset.")

    # Get SHAP values and feature names
    shap_vals = drug_shap.values[0]
    base_value = drug_shap.base_values[0]
    features = X_cols
    top_n = 10

    # Sort by absolute SHAP value
    sorted_indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    sorted_shap = shap_vals[sorted_indices]
    sorted_features = [features[i] for i in sorted_indices]

    # Custom plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_features[::-1], sorted_shap[::-1], color="skyblue")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Top {top_n} SHAP Contributions for {drug_name}")
    plt.xlabel("SHAP Value")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
