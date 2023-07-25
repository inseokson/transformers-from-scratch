from pathlib import Path, PosixPath
from typing import Union

import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(
        self,
        path: Union[PosixPath, str],
        source_column: str,
        target_column: str,
        *,
        bos_token: str = None,
        eos_token: str = None,
        add_bos_eos_token: bool = False,
    ):
        if isinstance(path, str):
            path = Path(path)

        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix in [".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"]:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"{str(path)} has not proper file extension")

        self.sources = df[source_column].to_numpy().flatten()
        self.targets = df[target_column].to_numpy().flatten()

        if add_bos_eos_token:
            self.bos_token = bos_token if bos_token is not None else "<bos>"
            self.eos_token = eos_token if eos_token is not None else "<eos>"

            self.sources = [self._add_bos_eos_token(sentence) for sentence in self.sources]
            self.targets = [self._add_bos_eos_token(sentence) for sentence in self.targets]

    def _add_bos_eos_token(self, sentence: str):
        if not sentence.startswith(self.bos_token):
            sentence = self.bos_token + sentence
        if not sentence.endswith(self.eos_token):
            sentence = sentence + self.eos_token

        return sentence

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx: int):
        return self.sources[idx], self.targets[idx]


def create_pad_mask(pad_mask):
    return pad_mask.unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(max_length):
    return (1 - torch.tril(torch.ones(max_length, max_length))).type(torch.ByteTensor)
