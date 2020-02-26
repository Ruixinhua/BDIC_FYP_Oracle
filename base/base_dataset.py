from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class BaseDataset(Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    implement the following three functions to create a subclass:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    """
    def __init__(self, data_root):
        """Initialize the class; save the options in the class

        Parameters:
            data_root -- the root directory that stores data
        """
        self.root_dir = data_root

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            it can be a tuple(by default) or a dictionary depend on training process
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass
