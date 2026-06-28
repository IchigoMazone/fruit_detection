from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader, random_split
from src.dataset.utils import transform_subset, status_split

class Dataset:

    def __init__(self, root, train_transform=None, test_transform=None, val_transform=None):
        self.root = root
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.val_transform = val_transform
        self.train_size = None
        self.test_size = None
        self.val_size = None

    @property
    def _get_data(self, train_transform=None, test_transform=None):

        train_data = datasets.MNIST(
            root=self.root,
            train=True,
            transform=train_transform,
            download=False
        )

        test_data = datasets.MNIST(
            root=self.root,
            train=False,
            transform=test_transform,
            download=False
        )

        return train_data, test_data
    
    def dataset_split(self, train=0.7, test=0, val=0):

        if 0 > train or train > 1:
            raise ValueError("train must be >= 0 and <= 1")
        
        if 0 > test or test > 1:
            raise ValueError("test must be >= 0 and <= 1")
        
        if 0 > val or val > 1:
            raise ValueError("val must be >= 0 and <= 1")
        
        data_train, data_test = self._get_data
        all_data = ConcatDataset([data_train, data_test])
        data_train, data_test, data_val = None, None, None
        length_data = len(all_data)
        option = status_split.INIT_DATA_CUR

        if test == 0 and val == 0:
            train_size = int(train * length_data)
            test_size = length_data - train_size

            if train_size + test_size != length_data:
                raise ValueError("train_size + test_size = 1")

            option = status_split.NOT_DATA_VAL
            data_train, data_test = random_split(
                all_data,
                [train_size, test_size]
            )

        elif val == 0: 
            train_size = int(train * length_data)
            test_size = int(test * length_data)
            val_size = length_data - train_size - test_size

            if train_size + test_size + val_size != length_data:
                raise ValueError("train_size + test_size + val_test = 1")

            option = status_split.FULL_DATA_INF
            data_train, data_test, data_val = random_split(
                all_data,
                [train_size, test_size, val_size]
            )
        else:
            raise ValueError("option already exists")
        
        return (
            transform_subset(data_train, self.train_transform),
            transform_subset(data_test, self.test_transform),
            transform_subset(data_val, self.val_transform) if option == status_split.FULL_DATA_INF else None
        )


if __name__ == "__main__":

    data = Dataset(
        root="./data",
    )

    train, test, val = data.dataset_split()

    train_loader = DataLoader(
        train, batch_size=32, shuffle=True
    )

    print(train.transform)
    print(test.transform)
    print(val.transform)

    

    
