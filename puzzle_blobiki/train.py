import click
import torch
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
@click.option("--learning-rate", type=float, default=1e-3, help="Learning rate")
@click.option("--hidden-size", default=128, help="Hidden size")
@click.option("--num-transformers", default=8, help="Number of transformers")
@click.option("--num-lstms", default=64, help="Number of LSTMs")
@click.option("--dropout", default=0.0, help="Dropout")
def main(strategy: str, device: str, batch_size: int, num_epochs: int, learning_rate: float, hidden_size: int,
         num_transformers: int, num_lstms: int, dropout: float) -> None:
    data = Data(strategy=Strategy[strategy], device=device)
    model = BlobNet(hidden_size=hidden_size, num_transformers=num_transformers, num_lstms=num_lstms, batch_size=batch_size,
                    dropout=dropout).to(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    print("=" * 30)
    print(f"Train: {len(data.train)} instances, mean: {data.train.mean: 2.4f}, std: {data.train.std: 2.4f}")
    print(f"Valid: {len(data.valid)} instances, mean: {data.valid.mean: 2.4f}, std: {data.valid.std: 2.4f}")
    print(f"Test: {len(data.test)} instances, mean: {data.test.mean: 2.4f}, std: {data.test.std: 2.4f}")
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
            optimizer.step()
            loss_history.append(sqrt(loss.item()))
        print(f"\tTrain RMSE: {mean(loss_history): 2.4f}")

        # Valid
        model.eval()
        with torch.no_grad():
            loss_history = []
            for x, y in tqdm(data.valid_loader(batch_size=batch_size), desc="Valid", leave=False):
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss_history.append(sqrt(loss.item()))
            print(f"\tValid RMSE: {mean(loss_history): 2.4f}")

    # Test
    loss_history = []
    for x, y in tqdm(data.test_loader(batch_size=batch_size), desc="Test", leave=False):
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss_history.append(sqrt(loss.item()))
    print(f"Final test RMSE: {mean(loss_history): 2.4f}")


if __name__ == "__main__":
    main()
