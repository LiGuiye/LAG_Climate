import numpy as np
from glob import glob
from utils.utils import normalize
from torchvision import transforms
from torch.utils.data import Dataset


class get_dataset(Dataset):
    def __init__(self, path, image_size, entire_mean, entire_std):
        self.root = path
        self.entire_mean = entire_mean
        self.entire_std = entire_std

        self.files = []
        if type(self.root) == list:
            for i in self.root:
                self.files += glob(f"{i}/*.npy")
        else:
            self.files = glob(f"{self.root}/*.npy")
        assert len(self.files), 'No data found.'

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
            f = glob(f"{self.root}/{index}.npy")[0]
        else:
            try:
                f = self.files[index]
            except:
                print("Given index doesn't exist, the first is used.")
                f = self.files[0]

        img = np.load(f).astype(np.float32)

        img = self.resize2tensor(img)
        img = normalize(img, self.entire_mean, self.entire_std)

        return img
