from matplotlib import pyplot as plt

from dl_dataset import Dataset


class Visualizer:
    def __init__(self, ds: Dataset):
        self.train_dataloader = ds.dataloader("train")
        self.train_ds = ds.train_ds

    def show_samples(self, num_samples: int = 5):
        for i in range(num_samples):
            img, caption = self.train_ds[i]
            plt.imshow(img)
            plt.title(caption)
            plt.show()
