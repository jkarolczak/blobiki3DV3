import click
import torch
import wandb
from numpy import mean, sqrt
from tqdm import tqdm

from src.data import Data, Strategy
from src.model import BlobNet


@click.command()
@click.option("--strategy", type=click.Choice(["BY_PUZZLE", "RANDOM"]), default="BY_PUZZLE",
              help="Strategy for data curation")
@click.option("--device", default="cpu", help="Device")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--num-epochs", default=10, help="Number of epochs")
@click.option("--learning-rate", type=float, default=1e-2, help="Learning rate")
@click.option("--hidden-size", default=128, help="Hidden size")
@click.option("--num-transformers", default=8, help="Number of transformers")
@click.option("--num-extractors", default=4, help="Number of feature extraction layers")
@click.option("--dropout", default=0.0, help="Dropout")
@click.option("--log", "log", flag_value=True, default=False, help="Log into wandb")
def main(strategy: str, device: str, batch_size: int, num_epochs: int, learning_rate: float, hidden_size: int,
         num_transformers: int, num_extractors: int, dropout: float, log: bool) -> None:
    data = Data(strategy=Strategy[strategy], device=device)
    model = BlobNet(hidden_size=hidden_size, num_transformers=num_transformers, num_extractors=num_extractors,
                    dropout=dropout).to(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    print("=" * 30)
    print(f"Train: {len(data.train)} instances, mean: {data.train.mean: 2.4f}, std: {data.train.std: 2.4f}")
    print(f"Valid: {len(data.valid)} instances, mean: {data.valid.mean: 2.4f}, std: {data.valid.std: 2.4f}")
    print(f"Test: {len(data.test)} instances, mean: {data.test.mean: 2.4f}, std: {data.test.std: 2.4f}")
    print("=" * 30)
    not log or wandb.init(
        project="rna-puzzles",
        config={
            "strategy": strategy, "device": device, "batch_size": batch_size, "num_epochs": num_epochs,
            "learning_rate": learning_rate, "hidden_size": hidden_size, "num_transformers": num_transformers,
            "num_extractors": num_extractors, "dropout": dropout,
            "train_size": len(data.train), "train_mean": data.train.mean, "train_std": data.train.std,
            "valid_size": len(data.valid), "valid_mean": data.valid.mean, "valid_std": data.valid.std,
            "test_size": len(data.test), "test_mean": data.test.mean, "test_std": data.test.std
        }
    )

    for epoch in range(num_epochs):
        print("=" * 30)
        print(f"Epoch {epoch + 1}")
        # Train
        model.train()
        loss_history = []
        for x, y in tqdm(data.train_loader(batch_size=batch_size), desc="Train", leave=False):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss_history.append(sqrt(loss.item()))
        print(f"\tTrain RMSE: {mean(loss_history): 2.4f}")
        not log or wandb.log({"train_rmse": mean(loss_history)}, step=epoch)

        # Valid
        model.eval()
        with torch.no_grad():
            loss_history = []
            for x, y in tqdm(data.valid_loader(batch_size=batch_size), desc="Valid", leave=False):
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss_history.append(sqrt(loss.item()))
            print(f"\tValid RMSE: {mean(loss_history): 2.4f}")
            not log or wandb.log({"valid_rmse": mean(loss_history)}, step=epoch)

    # Test
    loss_history = []
    for x, y in tqdm(data.test_loader(batch_size=batch_size), desc="Test", leave=False):
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss_history.append(sqrt(loss.item()))
    print(f"Final test RMSE: {mean(loss_history): 2.4f}")
    not log or wandb.log({"test_rmse": mean(loss_history)})
    wandb.finish()


if __name__ == "__main__":
    main()
