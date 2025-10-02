from matplotlib import pyplot as plt
import torch

from flicker_dataset import Flicker30kDataset


class Visualizer:
    def __init__(self, ds: Flicker30kDataset):
        self.ds = ds

    def show_random_image(self):
        dataloder = torch.utils.data.DataLoader(
            self.ds,
            batch_size=1,
            collate_fn=Flicker30kDataset.collate_fn,
        )
        for i, sample in enumerate(dataloder):
            title = "\n".join(sample["caption"])
            plt.imshow(sample["image"][0].permute(1, 2, 0))
            plt.title(title)
            plt.show()
            break

    def caption_length_distribution(self) -> plt.Figure:
        fig = plt.figure()
        caption_lengths = [len(caption) for caption in self.ds.captions]
        plt.hist(caption_lengths, bins=20)
        return fig
