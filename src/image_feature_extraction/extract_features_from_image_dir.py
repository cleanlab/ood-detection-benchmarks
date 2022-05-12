import numpy as np
import onnxruntime
from .image_utils import read_images_as_data_loader
from utils.time_utils import timefunc


@timefunc
def extract_features_from_image_dir(
    path_to_onnx: str, path_to_image_dir: str, batch_size: int = 32
) -> np.array:

    # read images
    data_loader, images = read_images_as_data_loader(path_to_image_dir, batch_size)

    print(f"Starting ONNX runtime engine...")
    print(f"  ONNX runtime device: {onnxruntime.get_device()}")

    ort_session = onnxruntime.InferenceSession(
        path_to_onnx, providers=["CUDAExecutionProvider"]
    )

    print(f"  ONNX runtime session providers: {ort_session.get_providers()}")

    input_name = ort_session.get_inputs()[0].name

    embeddings = []
    labels = []
    batches = len(data_loader)
    print("Extracting features...")
    for i, batch in enumerate(data_loader):

        # print progress of extract features
        print(f"  Batch {i+1} out of {batches}", end="\r")

        # prepare input
        X, y = batch
        X = X.numpy().astype(np.float32)
        y = y.numpy().astype(int)

        # run inference
        ort_outs = ort_session.run(None, input_feed={input_name: X})

        # append output
        embeddings.append(ort_outs[0])
        labels.append(y)

    return np.vstack(embeddings), np.hstack(labels), images


if __name__ == "__main__":
    pass
