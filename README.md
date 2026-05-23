# GAN Variants for Balancing Imbalanced Datasets
Credit Card Fraud Detection — Special Topics in Data Science, Spring 2025/2026

## What this project does
Implements and compares **Vanilla GAN**, **WGAN**, and **CGAN** to generate synthetic fraud transactions and fix the class imbalance problem (578:1 ratio). An MLP classifier is then trained and evaluated across four scenarios.

## Results

| Scenario    | Precision | Recall | F1     | AUC-ROC |
|-------------|-----------|--------|--------|---------|
| Original    | 0.7905    | 0.8469 | 0.8177 | 0.9790  |
| Vanilla GAN | 0.8316    | 0.8061 | 0.8187 | 0.9773  |
| WGAN        | 0.8889    | 0.7347 | 0.8045 | 0.9784  |
| CGAN        | 0.8851    | 0.7857 | 0.8324 | 0.9782  |

**CGAN achieves the best F1-score (0.832).**

## Dataset
`creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`.
