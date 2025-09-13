import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import shap
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
import xgboost as xgb
from shap import TreeExplainer
from shap import summary_plot
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    f1_score
)

def main():
    df = pd.read_csv("final.csv")
    df = df.drop(["optn_LE", "cavRef_LE", "optN_n_LGFE", "smiles"], axis=1)
    df = df.dropna()
    df["Risk"] = df["Risk"].replace({ "PR": "PR+CR+SR", "CR": "PR+CR+SR", "SR": "PR+CR+SR" })
    print(df.columns)
    print(len(df))

    le = LabelEncoder()

    df = df[df.Risk.isin(["KR", "NR"])]

    X_df = df.drop(["Ligand", "Risk", "TdP"], axis=1)
    X_cols = X_df.columns
    X = X_df.values
    Y = df["TdP"].astype(int).values
    # Y = le.fit_transform(df["Risk"].values)
    # Y = le.fit_transform(df["Risk"].replace({ "PR": "PR+CR+SR", "CR": "PR+CR+SR", "SR": "PR+CR+SR" }).values)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    best_params = {
        'max_depth': 9,
        'learning_rate': 0.0902094894773604,
        'subsample': 0.6814146548895265,
        'colsample_bytree': 0.7734010040924201,
        'reg_alpha': 0.19361805119494457,
        'reg_lambda': 1.0458362761910434
        }

    clf = xgb.XGBClassifier(
        n_estimators=193,  # You can tune this too if needed
        random_state=42,
        **best_params
    )

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    specificity = tn / (tn + fp)

    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions, average="macro")
    # Compute and print ROC AUC
    roc_auc = roc_auc_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)
    f1 = f1_score(y_test, predictions)


    print(f"Accuracy: {acc}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Sensitivity/recall: {recall}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"MCC: {mcc:.3f}")
    print(f"F1 Score: {f1:.3f}")


    # class_accuracies = []
    # print(np.unique(predictions))
    # print(le.classes_)
    # for class_ in np.unique(predictions):
    #     class_acc = np.mean(predictions[y_test == class_] == class_)
    #     class_accuracies.append(class_acc)

    # print(class_accuracies)

    # print(df[df["Ligand"].str.contains("Amiodarone")])

    amio_row = X_df.loc[df["Ligand"].str.contains("Amiodarone")].values

    explainer = shap.Explainer(clf, X, feature_names=X_cols)

    # Global explanation (beeswarm)
    shap_values_train = explainer(X_train)
    shap.plots.beeswarm(shap_values_train, show=True)

    # Explanation for Amiodarone row
    amio_index = df[df["Ligand"].str.contains("Amiodarone")].index[0]
    amio_row_values = X[amio_index:amio_index+1]
    shap_values_amio = explainer(amio_row_values)

    # Waterfall plot for Amiodarone
    shap.plots.waterfall(shap_values_amio[0], show=True)

    # Print for reference
    print(X_df.loc[df["Ligand"].str.contains("Amiodarone")])


    print(X_cols)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
