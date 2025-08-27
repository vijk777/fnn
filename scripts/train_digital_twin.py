#!/usr/bin/env python

"""
Train foundation readouts from a pre-trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
import torch.multiprocessing as mp
from fnn.data import load_training_data
from fnn.microns.build import network
from fnn.train.schedulers import CosineLr
from fnn.train.optimizers import SgdClip
from fnn.train.loaders import Batches
from fnn.train.objectives import NetworkLoss
from fnn import microns
from fnn.utils import logging
import torch
import yaml
import argparse

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_CONFIG = Path('/workspace/fnn/data/train_digital_twin/config.yaml')

def main(args):
    logger.info(f"Config file: {args.config}")

    # READ CONFIG
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # LOAD DATASET
    data_dir = config['data-source']['training'].get('directory', None)
    max_items = config['data-source']['training'].get('max_items', None)
    logger.info(f"Loading dataset from {data_dir}")
    dataset = load_training_data(data_dir, max_items)
    
    # PREPARE MULTIPROCESSING
    # Try forcing spawn method if it's not already set
    if mp.get_start_method(allow_none=True) != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
            logger.info("Successfully set start method to 'spawn'")
        except RuntimeError as e:
            logger.info(f"Could not set start method: {e}")

    # BUILD MODEL
    logger.info(f"Initializing model.")
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network(units=len(dataset.df.units.iloc[0][0])).to(device)

    # LOAD FOUNDATION MODEL PARAMETERS
    logger.info(f"Loading foundation core.")
    foundation_model, _ = microns.scan(**config['data-source']['foundation-core'])

    # TRANSFER CORE
    logger.info(f"Transferring foundation core to initialized model.")
    transfer_modules = ["core", "modulation.lstm"]
    for module in transfer_modules:
        _foundation_model = foundation_model.module(module)
        _model = model.module(module)

        # transfer parameters
        _model.load_state_dict(_foundation_model.state_dict())

        # freeze parameters
        _model.freeze(True)

    # BUILDING COMPONENTS FOR MODEL TRAINING
    logger.info(f"Building model training components.")
    scheduler = CosineLr(**config['scheduler'])
    optimizer = SgdClip(**config['optimizer'])
    loader = Batches(**config['loader'])
    objective = NetworkLoss(**config['objective'])

    # scheduler
    scheduler._init(epoch=0, cycle=0)

    # optimizer
    optimizer._init(scheduler=scheduler)

    # data loader
    loader._init(dataset=dataset)

    # training objective
    objective._init(network=model)

    # TRAIN NETWORK
    logger.info(f"Starting training.")
    epochs, metrics = [], []
    for epoch, info_dict in optimizer.optimize(
        loader=loader,
        objective=objective,
        parameters=model.named_parameters(),
        groups=None,
    ):
        epochs.append(epoch)
        metrics.append(info_dict)
    
    # SAVE DATA
    logger.info("Saving training metrics and model checkpoint.")

    save_dir = Path(config['save-state']['directory'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save metrics to CSV
    df = pd.DataFrame(metrics)
    df.insert(0, "epoch", epochs)
    metrics_csv = save_dir / config['save-state'].get("metrics_csv")
    df.to_csv(metrics_csv, index=False)
    logger.info(f"Training metrics written to {metrics_csv}")

    # 2) Save raw lists for resume or analysis
    metrics_pt = save_dir / config['save-state'].get("metrics_tensor")
    torch.save({"epochs": epochs, "metrics": metrics}, metrics_pt)
    logger.info(f"Raw metrics saved to {metrics_pt}")

    # 3) Save model weights
    torch.save(
        model.state_dict(),
        save_dir / config['save-state']['state_dict']
    )
    logger.info(f"Model state dict saved to {save_dir / config['save-state']['state_dict']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a readout model on neural data.")
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"Path to model config YAML (default: {DEFAULT_CONFIG})"
    )
    args = parser.parse_args()
    main(args)