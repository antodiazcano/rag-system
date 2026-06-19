"""Configuration of the project."""

from dataclasses import dataclass

# import torch


@dataclass
class ModelConfig:
    """Class to define the configuration of the model."""

    in_dim = 3
    out_dim = 1


@dataclass
class TrainingConfig:
    """Class to define the configuration of the training."""

    epochs = 10
    batch_size = 64
    lr = 1e-3
    # optimizer = torch.optim.Adam


@dataclass
class Config:
    """Main configuration class."""

    model = ModelConfig()
    training = TrainingConfig()


config = Config()
