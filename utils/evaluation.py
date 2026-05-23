"""Evaluation metrics and plotting utilities."""
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, precision_score, recall_score
)

COLORS = ["#A8C8E8", "#F4A8A8", "#A8D8A8", "#F4D8A8"]
STYLE = {
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#F5F5F5",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.color": "#FFFFFF",
    "grid.linewidth": 1.0,
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "text.color": "#444444",
    "axes.titlecolor": "#333333",
    "axes.labelcolor": "#555555",
}


def evaluate_classifier(model, loader, device, minority_class=1):
    model.eval()
    all_preds, all_probs, all_true = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.append(preds)
            all_probs.append(probs)
            all_true.append(y.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)[:, minority_class]

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def plot_class_distribution(y, title, save_path=None):
    with plt.rc_context(STYLE):
        classes, counts = np.unique(
            y.numpy() if torch.is_tensor(y) else y, return_counts=True
        )
        labels = ["Normal (0)", "Fraud (1)"]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, counts, color=[COLORS[0], COLORS[1]],
                      edgecolor="#FFFFFF", linewidth=0.8)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 200, f"{count:,}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Count")
        ax.set_title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()


def plot_generated_samples(samples, title, feature_names=None, save_path=None):
    """Plot distribution of first 8 features for generated vs real samples."""
    samples = samples.detach().cpu().numpy() if torch.is_tensor(samples) else samples
    n_features = min(8, samples.shape[1])
    fig, axes = plt.subplots(2, 4, figsize=(14, 5))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(title, color="#333333")
    for i, ax in enumerate(axes.flat):
        if i >= n_features:
            ax.axis("off")
            continue
        ax.hist(samples[:, i], bins=30, color=COLORS[0],
                edgecolor="#FFFFFF", linewidth=0.5)
        ax.set_title(feature_names[i] if feature_names else f"Feature {i+1}",
                     fontsize=9)
        ax.set_facecolor("#F5F5F5")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_metric_comparison(results, save_path=None):
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    scenarios = list(results.keys())
    n_scenarios = len(scenarios)
    width = 0.8 / n_scenarios

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(metrics))
        for i, scen in enumerate(scenarios):
            values = [results[scen][m] for m in metrics]
            ax.bar(x + i * width, values, width,
                   label=scen, color=COLORS[i % len(COLORS)],
                   edgecolor="#FFFFFF", linewidth=0.8)
        ax.set_xticks(x + width * (n_scenarios - 1) / 2)
        ax.set_xticklabels(metrics, rotation=15)
        ax.set_ylabel("Score")
        ax.set_title("Classifier Performance Across Scenarios")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()


def plot_confusion_matrices(results, save_path=None):
    n = len(results)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        fig.patch.set_facecolor("#FAFAFA")
        if n == 1:
            axes = [axes]
        for ax, (name, m) in zip(axes, results.items()):
            cm = m["confusion_matrix"]
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Normal", "Fraud"])
            ax.set_yticklabels(["Normal", "Fraud"])
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, f"{cm[r,c]:,}", ha="center",
                            va="center", fontsize=8,
                            color="white" if cm[r, c] > cm.max() / 2 else "black")
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()
