import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyroved as pv
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping handler to prevent overfitting."""

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class VAETrainer:
    def __init__(self, data_path: str, output_dir: str = "models", val_split: float = 0.2):
        """Initialize VAE trainer with data path and output directory."""
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.val_split = val_split

        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_data(self):
        """Load and preprocess data."""
        try:
            data = pd.read_csv(self.data_path)

            # Deduplicate and pivot data
            df_dedup = data.groupby(
                ['time', 'x', 'y', 'level'], as_index=False).mean()
            df_pivoted = df_dedup.set_index(
                ['time', 'y', 'x', 'level']).unstack('level')

            # Clean up column names
            df_pivoted.columns = [
                f'{level}-{var}' for var, level in df_pivoted.columns]
            df = df_pivoted.reset_index().drop(columns=['time', 'y', 'x'])

            # Remove rows with NaN values and normalize
            X = df[~np.isnan(df).any(axis=1)]
            X = normalize(X, axis=0)

            # Split data into train and validation sets
            X_train, X_val = train_test_split(
                X, test_size=self.val_split, random_state=42)

            # Convert to torch tensors
            train_data = torch.from_numpy(X_train).float().to(self.device)
            val_data = torch.from_numpy(X_val).float().to(self.device)

            self.train_loader = pv.utils.init_dataloader(
                train_data, batch_size=64)
            self.val_loader = pv.utils.init_dataloader(val_data, batch_size=64)
            self.in_dim = (X.shape[1],)

            logger.info(
                f"Data loaded successfully. Input dimension: {self.in_dim}")

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def validate(self, trainer, model):
        """Run validation step."""
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                loss = trainer.loss_function(model, batch)
                total_val_loss += loss.item()
        return total_val_loss / len(self.val_loader)

    def train(self, latent_dim: int, n_epochs: int = 100, patience: int = 5, min_delta: float = 0.001):
        """Train VAE model with specified latent dimension and early stopping."""
        save_path = self.output_dir / f"vae-{latent_dim}.pth"
        best_model_path = self.output_dir / f"vae-{latent_dim}_best.pth"

        try:
            # Initialize model
            vae = pv.models.iVAE(
                self.in_dim,
                latent_dim=latent_dim,
                invariances=None,
                seed=0
            ).to(self.device)

            trainer = pv.trainers.SVItrainer(vae)
            early_stopping = EarlyStopping(
                patience=patience, min_delta=min_delta)
            best_val_loss = float('inf')

            logger.info(f"Starting training for latent_dim={latent_dim}")

            # Training loop
            for epoch in range(n_epochs):
                # Training step
                train_loss = trainer.step(self.train_loader)

                # Validation step
                val_loss = self.validate(trainer, vae)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(vae.state_dict(), best_model_path)

                if (epoch + 1) % 5 == 0:  # Log every 5 epochs
                    logger.info(
                        f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                trainer.print_statistics()

                # Early stopping check
                if early_stopping(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # Load best model and save as final model
            vae.load_state_dict(torch.load(best_model_path))
            torch.save(vae.state_dict(), save_path)
            logger.info(f"Best model saved to {save_path}")

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise


def main():
    # Configuration
    data_path = './data/merged_data.csv'
    output_dir = "vae_models"
    latent_dims = [2, 4, 8, 16, 32]

    try:
        # Initialize trainer
        trainer = VAETrainer(data_path, output_dir, val_split=0.2)

        # Load and preprocess data
        trainer.load_data()

        # Train models with different latent dimensions
        for latent_dim in latent_dims:
            logger.info(f"Training model with latent_dim={latent_dim}")
            trainer.train(latent_dim, n_epochs=100,
                          patience=10, min_delta=0.001)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
