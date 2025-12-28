import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torchaudio
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification


class PkmnDataset(Dataset):
    """
    Le but est de recuperer les dossiers qu'on a melange nous meme et slit en train val test et d'en faire un Dataset Torch.
    On doit lui rajouter les __len__ et __getitem__ sinon ça marchera pas.

    Parametres :
    - data_dir (str) : chemin du dossier Data qu'on a créé avant (celui qui contient les dirs train val et test)
    - csv_path (str) : on donne le chemin du csv qui contient le maping pokémon : index (on l'a scrapé tout au début)
    - split (str) : comme on crée 3 dataset (train, val et test) on spécifie lequel on veut creer pour cette instance
    - feature_extractor (str) : feature extractor dont on va se servir (nous c'est celui de AudioSpectrogramTransformer)
    """

    def __init__(self, data_dir, csv_path, split, feature_extractor):
        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split)
        self.feature_extractor = feature_extractor
        self.csv_path = csv_path
        self.files = []
        self.labels = []

        pokedex = pd.read_csv(
            self.csv_path,
            usecols=["INT_numéro", "Pokémon"],
            dtype={"INT_numéro": int, "Pokémon": str},
            encoding="utf-8",
        )
        self.pokemon_index = dict(zip(pokedex["Pokémon"], pokedex["INT_numéro"] - 1))

        dir_cris = [
            dir
            for dir in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, dir))
        ]

        for pokemon in dir_cris:

            pokemon_dir = os.path.join(self.split_dir, pokemon)
            label = self.pokemon_index[pokemon]

            for cri in os.listdir(pokemon_dir):
                if ".mp3" in cri:
                    self.files.append(os.path.join(pokemon_dir, cri))
                    self.labels.append(label)

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        wf, sr = torchaudio.load(self.files[index])

        inputs = self.feature_extractor(
            wf.squeeze(0).numpy(), sampling_rate=sr, return_tensors="pt"
        )

        return inputs["input_values"].squeeze(0), self.labels[index]


class ModeleEntrainement(pl.LightningModule):
    """
    Ici on définit notre Lightning module pour l'entrainement de notre modele.

    Parametres :
    - nb_classes (int) : nombre de classes qu'on veut classifier (nous c'est 151 pokemons pour la 1G)
    - lr (float) : c'est le learning rate de notre modele (trop haut il surapprend, trop bas ça converge trop lentement)

    """

    def __init__(self, nb_classes, lr=1e-4):

        super().__init__()
        self.save_hyperparameters()

        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        self.nb_classes = nb_classes
        self.lr = lr

        for param in self.model.audio_spectrogram_transformer.parameters():
            param.requires_grad = False

        self.model.classifier = torch.nn.Linear(
            self.model.config.hidden_size, self.nb_classes
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):

        return self.model(x).logits

    def training_step(self, batch, batch_index):

        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_index):

        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=3
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class DataModule(pl.LightningDataModule):
    """
    Ici on gere tout ce qui est dataset et loaders pour notre modele.
    On pourrait le faire a la main plus bas mais bon tant qu'a faire autant le rendre portable ça peut tjr servir lol.

    Parametres :
    - data_dir (str) : chemin de notre data directory (celui qui est organise en train val test faqit dans le preprocessing)
    - csv_path (str) : le chemin de notre pokedex pour les labels et les index de nos pokemons
    - batch_size (int) : taille des batch qu'on envoie a notre modele.
    """

    def __init__(self, data_dir, csv_path, batch_size=8):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

    def setup(self, stage=None):

        self.train_dataset = PkmnDataset(
            self.data_dir, self.csv_path, "train", self.feature_extractor
        )
        self.val_dataset = PkmnDataset(
            self.data_dir, self.csv_path, "val", self.feature_extractor
        )
        self.test_dataset = PkmnDataset(
            self.data_dir, self.csv_path, "test", self.feature_extractor
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    BASE_DIR = os.getenv("BASE_DIR")
    DATA_DIR = os.path.join(BASE_DIR, "data/split_data")
    CSV_PATH = os.getenv("CSV_PATH")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")

    version = 0
    batch_size = 8
    epochs = 10
    patience = 5
    nb_classes = len(
        [
            pokemon
            for pokemon in os.listdir(TRAIN_DIR)
            if os.path.isdir(os.path.join(TRAIN_DIR, pokemon))
        ]
    )
    models_dir = os.path.join(BASE_DIR, "models", f"version_{version}")
    os.makedirs(models_dir, exist_ok=True)

    datamodule = DataModule(data_dir=DATA_DIR, csv_path=CSV_PATH, batch_size=batch_size)
    model = ModeleEntrainement(nb_classes=nb_classes)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=True),
        ModelCheckpoint(
            dirpath=models_dir,
            filename=f"version_{version}",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            verbose=True,
        ),
    ]

    trainer = pl.Trainer(
        default_root_dir=models_dir,
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model=model, datamodule=datamodule)
