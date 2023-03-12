import logging
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.loggers import get_experiment_logger
from pathlib import Path
from typing import Union
from loguru import logger
from .util import load_config


def train(
    config: Union[str, Path, dict, DictConfig],
) -> Trainer:
    """Train Anomalib model.

    Parameters
    ----------
    config : str, Path, dict
        Path to the configuration file or dictionary with the configuration.

    Returns
    -------
    Trainer
        Pytorch lightning trainer after training.
    """
    print(config)
    if isinstance(config, (str, Path)):
        logger.info("Load configuration file.")
        config = load_config(str(config))
    elif isinstance(config, dict):
        config = DictConfig(config)

    logger.info("Define Datamodule.")
    datamodule = get_datamodule(config)

    logger.info("Define Model.")
    model = get_model(config)

    logger.info("Define Trainer.")
    trainer = Trainer(
        **config.trainer,
        logger=get_experiment_logger(config),
        callbacks=get_callbacks(config),
    )

    logger.info("Training the model.")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.model.get("ckpt_path", None),
    )

    return trainer
