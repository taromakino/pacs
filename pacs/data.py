import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


N_CLASSES = 7
ENVS = [
    'art_painting',
    'cartoon',
    'photo',
    'sketch'
]
N_ENVS = len(ENVS) - 1 # Don't count test env


class PACSDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.transforms(Image.open(self.df.fpath.iloc[idx]).convert('RGB'))
        y = torch.tensor(self.df.y.iloc[idx], dtype=torch.long)
        e = torch.tensor(self.df.e.iloc[idx])
        if not torch.isnan(e).any():
            e = e.long()
        return x, y, e


def subsample(rng, df, n_eval_examples):
    if len(df) < n_eval_examples:
        return df
    else:
        idxs = rng.choice(len(df), n_eval_examples, replace=False)
        return df.iloc[idxs]


def make_data(test_env, train_ratio, batch_size, eval_batch_size, n_workers, n_eval_examples):
    rng = np.random.RandomState(0)
    trainval_envs = [env for env in ENVS if env != test_env]
    dpath = os.path.join(os.environ['DATA_DPATH'], 'PACS')

    df_trainval = {
        'fpath': [],
        'y': [],
        'e': []
    }
    for e_idx, e_name in enumerate(trainval_envs):
        classes = os.listdir(os.path.join(dpath, e_name))
        for y_idx, y_name in enumerate(classes):
            fnames = os.listdir(os.path.join(dpath, e_name, y_name))
            fpaths = [os.path.join(dpath, e_name, y_name, fname) for fname in fnames]
            df_trainval['fpath'] += fpaths
            df_trainval['y'] += [y_idx] * len(fpaths)
            df_trainval['e'] += [e_idx] * len(fpaths)
    df_trainval = pd.DataFrame(df_trainval).sample(frac=1)

    df_test = {
        'fpath': [],
        'y': [],
        'e': []
    }
    classes = os.listdir(os.path.join(dpath, test_env))
    for y_idx, y_name in enumerate(classes):
        fnames = os.listdir(os.path.join(dpath, test_env, y_name))
        fpaths = [os.path.join(dpath, test_env, y_name, fname) for fname in fnames]
        df_test['fpath'] += fpaths
        df_test['y'] += [y_idx] * len(fpaths)
        df_test['e'] += [float('nan')] * len(fpaths)
    df_test = pd.DataFrame(df_test).sample(frac=1)

    train_idxs = rng.choice(len(df_trainval), int(train_ratio * len(df_trainval)), replace=False)
    val_idxs = np.setdiff1d(np.arange(len(df_trainval)), train_idxs)

    df_train = df_trainval.iloc[train_idxs]
    df_val = df_trainval.iloc[val_idxs]

    if n_eval_examples is not None:
        df_val = subsample(rng, df_val, n_eval_examples)
        df_test = subsample(rng, df_test, n_eval_examples)

    data_train = DataLoader(PACSDataset(df_train), shuffle=True, pin_memory=True, batch_size=batch_size, num_workers=n_workers)
    data_val = DataLoader(PACSDataset(df_val), pin_memory=True, batch_size=eval_batch_size, num_workers=n_workers)
    data_test = DataLoader(PACSDataset(df_test), pin_memory=True, batch_size=eval_batch_size, num_workers=n_workers)
    return data_train, data_val, data_test