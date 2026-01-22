import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def get_metrics(model, data_loader, protein_graph):
    y_pred = []
    y_true = []

    # 1. Collect Data
    with torch.no_grad():  # Good practice to disable gradients here
        for x in data_loader:
            # Predict
            out = model(protein_graph, x).detach().cpu().numpy()
            y_pred.append(out)
            y_true.append(x.y.detach().cpu().numpy())

    # 2. Concatenate and FLATTEN (The Fix)
    y_pred = np.concatenate(y_pred).ravel()  # Force 1D array
    y_true = np.concatenate(y_true).ravel()  # Force 1D array

    # 3. Calculate Correct MSE
    mse = np.mean((y_pred - y_true) ** 2)
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))

    return mse, r2, y_pred, y_true


def plot_predictions(model, loaders, protein_graph, title):
    _, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for idx, loader in enumerate(loaders):
        mse, r2, y_pred, y_true = get_metrics(model, loader, protein_graph)

        ax[idx].scatter(y_pred, y_true, alpha=0.5)

        # Add a DIAGONAL line (Perfect Prediction) - Much more useful than the mean line
        m_min, m_max = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax[idx].plot([m_min, m_max], [m_min, m_max], "k--", label="Perfect Fit")
        ax[idx].set_aspect("equal", "box")
        if idx == 1:
            ax[idx].set_xlabel("Predicted pIC50")
        if idx == 0:
            ax[idx].set_ylabel("True pIC50")
        ax[idx].set_title(f"MSE: {mse:.4f} | R2: {r2:.4f}")
        ax[idx].legend()
        ax[idx].grid(True, alpha=0.3)
    plt.suptitle(title)
    plt.show()


def plot_xgboost_predictions(
    pred_train_mse_tuple, pred_val_mse_tuple, pred_test_mse_tuple
):
    _, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for idx, (y_pred, y_true, mse) in enumerate(
        [pred_train_mse_tuple, pred_val_mse_tuple, pred_test_mse_tuple]
    ):
        ax[idx].scatter(y_pred, y_true, alpha=0.5)

        # Add a DIAGONAL line (Perfect Prediction) - Much more useful than the mean line
        m_min, m_max = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax[idx].plot([m_min, m_max], [m_min, m_max], "k--", label="Perfect Fit")
        r2 = 1 - (
            np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        )

        if idx == 1:
            ax[idx].set_xlabel("Predicted pIC50")
        if idx == 0:
            ax[idx].set_ylabel("True pIC50")
        ax[idx].set_title(f"MSE: {mse:.4f} | R2: {r2:.4f}")
        ax[idx].legend()
        ax[idx].grid(True, alpha=0.3)

    plt.suptitle("XGBoost Predictions")
    plt.show()


def plot_xgb_loss(xgb_model):
    # 4. Extract Results
    results = xgb_model.evals_result()
    epochs = len(results["validation_0"]["rmse"])
    x_axis = range(0, epochs)

    # 5. Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, results["validation_0"]["rmse"], label="Train")
    ax.plot(x_axis, results["validation_1"]["rmse"], label="Validation")
    ax.plot(x_axis, results["validation_2"]["rmse"], label="Test")
    ax.legend()
    plt.ylabel("RMSE")
    plt.xlabel("n_estimators")
    plt.title("XGBoost Loss vs. n_estimators")
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Best Iteration: {xgb_model.best_iteration}")
    print(f"Best Score (Validation MSE): {xgb_model.best_score:.4f}")


def plot_loss(csv_path):
    df = pd.read_csv(csv_path)
    fix, ax = plt.subplots(figsize=(10, 5))

    df.plot(
        x="step",
        y="train_loss",
        title="Training Loss",
        ax=ax,
        label="Training Loss",
        color="blue",
        legend=True,
        style="-",
    )
    df.dropna(subset=["val_loss"]).plot(
        x="step",
        y="val_loss",
        title="Validation Loss",
        ax=ax,
        label="Validation Loss",
        color="red",
        legend=True,
        style="-o",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    plt.show()
