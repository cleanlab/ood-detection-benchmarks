from torchvision.datasets import ImageFolder
import torch
from utils.time_utils import timefunc
from torch.utils.data import DataLoader
from torchvision import transforms


@timefunc
def init_data_loader_from_image_folder(
    data_path, transform, batch_size=64, shuffle=False
):
    images = ImageFolder(data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        images, batch_size=batch_size, shuffle=shuffle
    )
    return data_loader, images


@timefunc
def read_images_as_data_loader(path_to_image_dir: str, batch_size: int) -> DataLoader:

    transform = transforms.Compose(
        [
            transforms.Resize(size=32),
            transforms.CenterCrop(size=32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    kwargs = {"transform": transform, "batch_size": batch_size, "shuffle": False}

    data_loader, images = init_data_loader_from_image_folder(
        path_to_image_dir, **kwargs
    )

    return data_loader, images
