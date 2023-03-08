import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from typing import Union, Any, List

from anomalib.data.utils import get_image_filenames
from anomalib.deploy import Inferencer
from anomalib.post_processing import ImageResult
from importlib import import_module
from .util import load_config


def get_inferencer(
    config: Union[str, Path, DictConfig],
    model_source: Path,
    meta_data_path: Path = None,
    device: str = "auto",
) -> Inferencer:
    extension = model_source.suffix
    inferencer: Inferencer
    module = import_module("anomalib.deploy")
    if extension == ".ckpt":
        torch_inferencer = getattr(module, "TorchInferencer")
        inferencer = torch_inferencer(
            config=config,
            model_source=model_source,
            meta_data_path=meta_data_path,
            device=device,
        )
    elif extension in (".onnx", ".bin", ".xml"):
        openvino_inferencer = getattr(module, "OpenVINOInferencer")
        inferencer = openvino_inferencer(
            config=config,
            path=model_source,
            meta_data_path=meta_data_path,
        )
    else:
        raise ValueError(
            "Model extension is not supported. Torch Inferencer exptects a "
            ".ckpt file, OpenVINO Inferencer expects either .onnx, .bin or "
            f".xml file. Got {extension}"
        )

    return inferencer


class InferenceModel:
    """Homogenize config and parameter inputs of TorchInferencer"""

    def __init__(
        self,
        config: Union[str, Path, DictConfig] = None,
        model_source: str = None,
        meta_data_path: str = None,
        accelerator: str = None,
    ):
        if isinstance(config, (str, Path)):
            config = load_config(config)
        config = self._update_config(
            config,
            model_source,
            meta_data_path,
            accelerator,
        )
        self._validate_config(config)

        self.model = get_inferencer(
            config=config,
            model_source=config.inferencer.model_source,
            meta_data_path=config.inferencer.meta_data_path,
            device=config.inferencer.accelerator,
        )

    @staticmethod
    def _validate_config(config):
        msg = "{key} must be specified in config.{group} or as parameter {param}"
        if "model_source" not in config.inferencer:
            raise KeyError(
                msg.format(group="inferencer", key="model_source", param="model_source")
            )
        if "accelerator" not in config.inferencer:
            raise KeyError(
                msg.format(group="inferencer", key="accelerator", param="accelerator")
            )

    @staticmethod
    def _update_config(
        config,
        model_source,
        meta_data_path,
        accelerator,
    ):
        if config is None:
            config = DictConfig({})

        if "inferencer" not in config:
            config.inferencer = {}

        if model_source is not None:
            config.inferencer.model_source = model_source
        if meta_data_path is not None:
            config.inferencer.meta_data_path = meta_data_path
        if accelerator is not None:
            config.inferencer.accelerator = accelerator

        return config

    def __call__(
        self,
        image: Union[str, Path, np.ndarray],
        meta_data: dict[str, Any] = None,
    ) -> List[ImageResult]:
        return self.predict(image=image, meta_data=meta_data)

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        meta_data: dict[str, Any] = None,
    ) -> List[ImageResult]:
        if isinstance(image, np.ndarray):
            return self._predict_array(image, meta_data)
        else:
            return self._predict_file(image, meta_data)

    def _predict_file(
        self,
        image: Union[str, Path, np.ndarray],
        meta_data: dict[str, Any] = None,
    ):
        image = get_image_filenames(path=image)
        if len(image) == 0:
            raise
        if len(image) == 1:
            self.model.predict(image=image[0], meta_data=meta_data)
        return [self.model.predict(image=i, meta_data=meta_data) for i in image]

    def _predict_array(
        self,
        image: Union[str, Path, np.ndarray],
        meta_data: dict[str, Any] = None,
    ):
        if not 3 <= image.ndim <= 4:
            raise
        if image.ndim == 3:
            return self.model.predict(image=image, meta_data=meta_data)
        return [self.model.predict(image=i, meta_data=meta_data) for i in image]
