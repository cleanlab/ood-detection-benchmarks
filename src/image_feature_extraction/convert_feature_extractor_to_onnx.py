import torch
from hydra import initialize_config_dir, compose
from pathlib import Path
from src.image_feature_extraction.feature_extractor import LitFeatureExtractor
from src.utils.time_utils import timefunc


@timefunc
def convert_feature_extractor_to_onnx():

    # init Hydra compose config
    CONFIG_PATH = Path("./configs").resolve()
    with initialize_config_dir(
        config_dir=str(CONFIG_PATH)
    ):  # absolute path to config file
        cfg = compose(config_name="default.yaml")

    file_path = cfg.feature_extractor.path_to_onnx

    model = LitFeatureExtractor()
    input_sample = torch.randn((1, 3, 32, 32))
    model.to_onnx(
        file_path,
        input_sample,
        export_params=True,
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    return model


if __name__ == "__main__":
    convert_feature_extractor_to_onnx()
