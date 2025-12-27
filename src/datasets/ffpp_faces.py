from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class FFPPFacesDataset(Dataset):
    """
    Dataset de caras recortadas de FF++ usando el CSV generado
    en data/processed/ffpp/ffpp_images_metadata.csv
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",  # "train", "val" o "test"
        transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.transform = transform

        df = pd.read_csv(self.csv_path)
        # Filtrar por split
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No hay filas para split={split} en {csv_path}")

        print(
            f"[FFPPFacesDataset] {split}: {len(self.df)} imágenes "
            f"({self.df['class_name'].value_counts().to_dict()})"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["file_path"])
        label = int(row["label"])        # 0 = real, 1 = fake

        # Abrir imagen
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return img, label_tensor
