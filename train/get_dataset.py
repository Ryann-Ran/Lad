from dataset import ABDOMENCT1KDataset
from torch.utils.data import WeightedRandomSampler


def get_dataset(cfg):
    if cfg.dataset.name == 'ABDOMENCT1K':
        train_dataset = ABDOMENCT1KDataset(root_dir=cfg.dataset.root_dir)
        val_dataset = ABDOMENCT1KDataset(root_dir=cfg.dataset.root_dir)
        print("train_dataset's size = ", len(train_dataset), "  val_dataset's size = ", len(val_dataset))
        sampler = None
        return train_dataset, val_dataset, sampler

    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
