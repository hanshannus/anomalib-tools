import logging
from pytorch_lightning import Trainer
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger
from pathlib import Path
from typing import Union
from .util import load_config

logger = logging.getLogger("anomalib")


def train(
    config_path: Union[str, Path],
) -> Trainer:
    """Train Anomalib model.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    Trainer
        Pytorch lightning trainer after training.
    """
    config_path = str(config_path)

    configure_logger(level="INFO")

    logger.info("Load configuration file.")
    config = load_config(config_path)

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
