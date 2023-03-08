import logging
from pytorch_lightning import Trainer
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger
from .util import load_config

logger = logging.getLogger("anomalib")


def train(
    config_path: str,
    data_dir: str = None,
    output_dir: str = None,
    ckpt_path: str = None,
    max_epochs: int = None,
) -> str:
    """Train Anomalib model.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    data_dir : str, optional
        Path to data root directory, by default None
    output_dir : str, optional
        Path to output data directory, by default None
    ckpt_path : str, optional
        Path to model checkpoint to load, by default None
    max_epochs : int, optional
        Maximum number of epochs, by default None

    Returns
    -------
    str
        Path to the best checkpoint.
    """
    configure_logger(level="INFO")

    logger.info("Load configuration file.")
    config = load_config(config_path)

    logger.info("Update config with input parameters.")
    if data_dir is not None:
        config.dataset.path = data_dir
    if output_dir is not None:
        config.project.path = output_dir
    if max_epochs is not None:
        config.trainer.max_epochs = max_epochs

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
        ckpt_path=ckpt_path,
    )

    best_model = trainer.checkpoint_callback.best_model_path
    logger.info(f"Path to best checkpoint: {best_model}")

    return best_model
