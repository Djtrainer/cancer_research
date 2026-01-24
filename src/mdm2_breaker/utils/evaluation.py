import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score


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
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

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
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_xgboost_predictions(
    pred_train_mse_tuple, pred_val_mse_tuple, pred_test_mse_tuple
):
    _, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for idx, (y_pred, y_true, _) in enumerate(
        [pred_train_mse_tuple, pred_val_mse_tuple, pred_test_mse_tuple]
    ):
        ax[idx].scatter(y_pred, y_true, alpha=0.5)

        # Add a DIAGONAL line (Perfect Prediction) - Much more useful than the mean line
        m_min, m_max = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax[idx].plot([m_min, m_max], [m_min, m_max], "k--", label="Perfect Fit")
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
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
    
    # Create the main figure and the primary axis (Left Y-axis for Loss)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1. Plot Training Loss (Left Axis)
    train_df = df.dropna(subset=["train_loss"])
    ax1.plot(train_df["step"], train_df["train_loss"], label="Training Loss", color="blue", linestyle="-", alpha=0.7)
    
    # 2. Plot Validation Loss (Left Axis)
    val_df = df.dropna(subset=["val_loss"])
    ax1.plot(val_df["step"], val_df["val_loss"], label="Validation Loss", color="red", marker="o", linestyle="-")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss (MSE)", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.grid(True, alpha=0.3)

    if "lr" in df.columns:
        lr_df = df.dropna(subset=["lr"])
    
        # 3. Create a TWIN axis (Right Y-axis for Learning Rate)
        ax2 = ax1.twinx()
        # Plot LR in green on the right axis
        ax2.plot(lr_df["step"], lr_df["lr"], label="Learning Rate", color="green", linestyle="--", alpha=0.8)
        
        ax2.set_ylabel("Learning Rate", color="green")
        ax2.tick_params(axis='y', labelcolor="green")

        # Set Title
        plt.title("Training Dynamics: Loss & Learning Rate")
        
        # 4. Combined Legend (Merge handles from both axes)
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center")

    plt.show()