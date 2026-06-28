from torchvision import datasets
from torch.utils.data import Dataset
from enum import Enum

class transform_subset(Dataset):

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
    
class status_split(Enum):

    INIT_DATA_CUR = 0
    NOT_DATA_VAL = 1
    FULL_DATA_INF = 2

def download_mnist(root: str):

    datasets.MNIST(
        root=root,
        train=True,
        transform=None,
        download=True
    )

    datasets.MNIST(
        root=root,
        train=False,
        transform=None,
        download=True
    )


        
