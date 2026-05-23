"""Load Credit Card Fraud dataset and prepare loaders."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_fraud(csv_path="data/creditcard.csv"):
    """Load and preprocess the Credit Card Fraud dataset.

    Returns train/test splits as tensors, already scaled.
    Minority class = 1 (fraud), Majority class = 0 (normal).
    """
    df = pd.read_csv(csv_path)

    # Drop 'Time', keep 'Amount' + V1-V28 + 'Class'
    df = df.drop(columns=["Time"])

    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = df["Class"].values.astype(np.int64)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split (stratified to preserve imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return (
        torch.tensor(X_train), torch.tensor(y_train),
        torch.tensor(X_test),  torch.tensor(y_test),
    )


def get_minority_loader(x, y, minority_class=1, batch_size=64):
    """Loader containing only minority class samples."""
    idx = (y == minority_class).nonzero(as_tuple=True)[0]
    ds = TensorDataset(x[idx], y[idx])
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def get_classifier_loader(x, y, batch_size=256, shuffle=True):
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)