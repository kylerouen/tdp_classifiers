import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import f1_score, matthews_corrcoef
import random
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # No Sigmoid here
        )

    def forward(self, x):
        return self.model(x)

def main():
    set_seed(42)  # Ensures reproducibility
    # Load and preprocess data
    df = pd.read_csv("final.csv")
    df = df.drop(["optn_LE", "cavRef_LE", "optN_n_LGFE", "smiles"], axis=1)
    df = df.dropna()
    df["Risk"] = df["Risk"].replace({"PR": "PR+CR+SR", "CR": "PR+CR+SR", "SR": "PR+CR+SR"})
    df = df[df["Risk"].isin(["KR", "NR"])]

    X_df = df.drop(["Ligand", "Risk", "TdP"], axis=1)
    X = X_df.values.astype(np.float32)
    Y = df["TdP"].astype(np.float32).values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).unsqueeze(1)

    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Optional: compute class imbalance weight
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)

    # Model, loss, optimizer
    model = SimpleNN(input_dim=X.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    # Early stopping setup
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    n_epochs = 100

    for epoch in range(n_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test_tensor)
            val_loss = criterion(val_preds, y_test_tensor)

        print(f"Epoch [{epoch+1}/{n_epochs}], Validation Loss: {val_loss.item():.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))

    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).numpy()
        preds_binary = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds_binary)
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds_binary)
        mcc = matthews_corrcoef(y_test, preds_binary)

        print(f"\nTest Accuracy:     {acc:.3f}")
        print(f"Test ROC AUC:      {auc:.3f}")
        print(f"Test F1 Score:     {f1:.3f}")
        print(f"Test MCC:          {mcc:.3f}")

    # SHAP explanation
    def predict_fn(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(model(x_tensor)).numpy()

    background = X_train[:100]
    explainer = shap.KernelExplainer(predict_fn, background, link="logit")
    X_subset = X_test[:100]
    shap_values = explainer(X_subset)

    if shap_values.values.shape[-1] == 1:
        shap_values.values = shap_values.values.squeeze(-1)

    shap_values.data = X_subset
    shap_values.feature_names = X_df.columns.tolist()

    shap.plots.beeswarm(shap_values, show=True)

if __name__ == "__main__":
    main()
