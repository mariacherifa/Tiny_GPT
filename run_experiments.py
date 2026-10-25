#######################################################################################
#                             Script for running experiments
#######################################################################################
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.data import load_data, build_vocab, data_split
from src.model import TinyGPT
from src.train import Train


def run_experiment(config_path):

    # 1. Read experiment configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    torch.manual_seed(config["seed"])

    # 2. Create a unique experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config["experiment_name"]

    output_dir = Path("experiments") / (
        f"{experiment_name}_{timestamp}"
    )
    output_dir.mkdir(parents=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 3. Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    # 4. Load data
    text = load_data(
        config["data_path"],
        max_chars=config["max_chars"]
    )

    vocab, vocab_size, itos, stoi = build_vocab(text)

    #  data_split 
    dataset, train_data, val_data = data_split(text)

    # 5. Initialize model
    model = TinyGPT(
        config["n_embd"],
        vocab_size,
        config["num_heads"],
        config["block_size"],
        config["num_layers"],
        config["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"]
    )

    # 6. Initialize trainer
    trainer = Train(
        model,
        train_data,
        val_data,
        config["n_embd"],
        config["batch_size"],
        config["block_size"],
        config["num_epochs"],
        optimizer,
        config["steps_per_epoch"],
        config["val_every"],
        device
    )

    # 7. Train
    history = trainer.fit()

    # 8. Save metrics
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch, train_loss, val_loss in zip(
            history["Epochs"],
            history["train_losses"],
            history["val_losses"]
        ):
            writer.writerow([epoch, train_loss, val_loss])

    # 9. Save final model checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "itos": itos,
        "stoi": stoi
    }, output_dir / "last_model.pt")

    # 10. Save learning curves
    plt.figure(figsize=(10, 5))

    plt.plot(
        history["Epochs"],
        history["train_losses"],
        label="Train"
    )

    plt.plot(
        history["Epochs"],
        history["val_losses"],
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=160,
        bbox_inches="tight"
    )
    plt.close()

    print("Experiment saved to:", output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    args = parser.parse_args()

    run_experiment(args.config)
