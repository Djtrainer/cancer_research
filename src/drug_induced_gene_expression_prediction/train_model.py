import os
import argparse

import numpy as np
import random
import torch
import wandb
import yaml

from drug_induced_gene_expression_prediction import run_training_pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROCESSED_DATA_PATH = os.path.join(
    "..", "..", "data", "processed", "drug_induced_gene_expression_prediction"
)

BATCH_SIZE = 64

# Random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a CVAE model for drug-induced gene expression prediction."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sweep", "train"],
        default="train",
        help="Mode to run: 'sweep' for hyperparameter sweep, 'train' for training a single model.",
    )
    args = parser.parse_args()
    if args.mode == "sweep":
        # Start hyperparameter sweep
        with open(
            os.path.join(
                "config",
                "drug_induced_gene_expression_prediction",
                "sweep_parameters.yaml",
            ),
            "r",
        ) as file:
            config = yaml.safe_load(file)

        sweep_id = wandb.sweep(
            config, project="drug-induced-gene-expression-prediction"
        )

        def sweep_agent_func():
            run_training_pipeline(
                config=None,
                processed_data_path=PROCESSED_DATA_PATH,
                batch_size=BATCH_SIZE,
                device=DEVICE,
                is_sweep=True,
            )

        wandb.agent(sweep_id, function=sweep_agent_func, count=500)

    else:
        # Train a single model
        with open(
            os.path.join(
                "config",
                "drug_induced_gene_expression_prediction",
                "model_parameters.yaml",
            ),
            "r",
        ) as file:
            config = yaml.safe_load(file)

        run_training_pipeline(
            config,
            processed_data_path=PROCESSED_DATA_PATH,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            is_sweep=False,
        )
