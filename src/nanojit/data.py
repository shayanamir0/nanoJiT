import os
import tarfile
import urllib.request
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def loader(img_size=256, batch_size=12):
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    data_dir = "./data"
    dataset_path = os.path.join(data_dir, "imagenette2-320/train")
    if not os.path.exists(os.path.join(data_dir, "imagenette2-320")):
        os.makedirs(data_dir, exist_ok=True)
        print(">>Downloading ImageNette...")
        urllib.request.urlretrieve(url, os.path.join(data_dir, "imagenette.tgz"))
        with tarfile.open(os.path.join(data_dir, "imagenette.tgz"), "r:gz") as tar:
            tar.extractall(path=data_dir)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)