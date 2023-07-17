from pathlib import Path, PosixPath
from typing import Union

import pandas as pd
from torch.utils.data import Dataset


class TabularDatasetForTranslator(Dataset):
    def __init__(
        self,
        path: Union[PosixPath, str],
        from_lang_col: str,
        to_lang_col: str,
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

        self.from_lang = df[from_lang_col].to_numpy().flatten()
        self.to_lang = df[to_lang_col].to_numpy().flatten()

        if add_bos_eos_token:
            self.bos_token = bos_token if bos_token is not None else "<bos>"
            self.eos_token = eos_token if eos_token is not None else "<eos>"

            self.from_lang = [self._add_bos_eos_token(sentence) for sentence in self.from_lang]
            self.to_lang = [self._add_bos_eos_token(sentence) for sentence in self.to_lang]

    def _add_bos_eos_token(self, sentence: str):
        if not sentence.startswith(self.bos_token):
            sentence = self.bos_token + sentence
        if not sentence.endswith(self.eos_token):
            sentence = sentence + self.eos_token

        return sentence

    def __len__(self):
        return len(self.from_lang)

    def __getitem__(self, idx: int):
        return self.from_lang[idx], self.to_lang[idx]
