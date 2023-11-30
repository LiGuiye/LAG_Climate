from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.utils import normalize


class get_dataset(Dataset):
    def __init__(self, path, image_size, entire_mean, entire_std, image_type):
        """
        Args:
            image_type: 'Solar', 'Wind'
        """
        self.root = path
        self.type = image_type
        self.entire_mean = entire_mean
        self.entire_std = entire_std

        self.files = []
        if type(path) == list:
            for i in path:
                self.files += glob(i + '/*.npy')
            assert self.files is not None, 'No data found.'
            self.suffix = 'npy'
        else:
            self.files = glob(self.root + '/*.tif')
            self.suffix = 'tif'
            if not self.files:  # in case images are in npy format
                self.files = glob(self.root + '/*.npy')
                assert self.files is not None, 'No data found.'
                self.suffix = 'npy'

        self.resize2tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert from HWC to CHW
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
            ]
        )

        self.image_size = image_size
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if isinstance(index, str):
            f = glob(self.root + '/' + index + '.' + self.suffix)[0]
        else:
            try:
                f = self.files[index]
            except:
                print("Given index doesn't exist, the first is used.")
                f = self.files[0]

        if self.suffix == 'tif':
            img = Image.open(f).astype(np.float32)
        else:
            img = np.load(f).astype(np.float32)

        img = self.resize2tensor(img)
        img = normalize(img, self.entire_mean, self.entire_std)

        return img
