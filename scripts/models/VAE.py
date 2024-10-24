import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyroved as pv
import torch
from sklearn.preprocessing import normalize

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VAETrainer:
    def __init__(self, data_path: str, output_dir: str = "models"):
        """Initialize VAE trainer with data path and output directory."""
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

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

            # Convert to torch tensor
            train_data = torch.from_numpy(X).float().to(self.device)
            self.train_loader = pv.utils.init_dataloader(
                train_data, batch_size=64)
            self.in_dim = (X.shape[1],)

            logger.info(
                f"Data loaded successfully. Input dimension: {self.in_dim}")

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def train(self, latent_dim: int, n_epochs: int = 100):
        """Train VAE model with specified latent dimension."""
        save_path = self.output_dir / f"vae-{latent_dim}.pth"

        try:
            # Initialize model
            vae = pv.models.iVAE(
                self.in_dim,
                latent_dim=latent_dim,
                invariances=None,
                seed=0
            ).to(self.device)

            trainer = pv.trainers.SVItrainer(vae)

            logger.info(f"Starting training for latent_dim={latent_dim}")

            # Training loop
            for epoch in range(n_epochs):
                loss = trainer.step(self.train_loader)
                if (epoch + 1) % 10 == 0:  # Log every 10 epochs
                    logger.info(
                        f"Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}")
                trainer.print_statistics()

            # Save model
            torch.save(vae.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

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
        trainer = VAETrainer(data_path, output_dir)

        # Load and preprocess data
        trainer.load_data()

        # Train models with different latent dimensions
        for latent_dim in latent_dims:
            logger.info(f"Training model with latent_dim={latent_dim}")
            trainer.train(latent_dim)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
