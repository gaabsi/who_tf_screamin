import os
import shutil
import warnings

import numpy as np
import torch
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning)


class AudioPreprocessing:
    """
    Classe tres courte pour le preprocessing de mes fichiers audio.
    Comme les audios de base ont l'air plutot propres (scrapés via mes petites mimines) peu de preprocessing a faire.

    Les principales composantes sont la standardisation des audios : tous les audios feront la taille de l'audio le plus long.
    Les audios seront splités en 3 dir que sont train val et test pour faciliter l'entrainement de notre modele.
    """

    def __init__(self, dir_input, dir_output, train_size, val_size, sr_fixe):
        self.dir_input = dir_input
        self.dir_output = dir_output
        self.train_size = train_size
        self.val_size = val_size
        self.sr_fixe = sr_fixe
        self.durees = []
        if train_size is not None and val_size is not None:
            self.test_size = 1 - (train_size + val_size)
        else:
            self.test_size = None

    def get_duree_audio(self, chemin_audio):
        """
        Calcule la durée d'un audio (en secondes).

        Parametres :
        - chemin_audio (str) : chemin de l'audio.

        Output :
        - duree (float) : durée de l'audio (en secondes).
        - wf (torch.Tensor) : waveform de l'audio (pour utilisation plus tard, evite de def trop de truc)
        - sr (int) : sample rate de l'audio
        """

        wf, sr = torchaudio.load(chemin_audio)
        duree = wf.shape[1] / sr

        return duree, wf, sr

    def get_duree_max(self, directory=None):
        """
        Donne la durée de l'audio le plus long d'un directory.

        Parametres :
        - directory (str) : chemin du directory qui contient les audios.
            En vrai c'est optionnel, je l'ai mit au cas ou mais ça m'etonnerait que je m'en serve de maniere isolée.
            Pour ça que j'ai mit None en default

        Output :
        - duree_max (float) : durée de l'audio le plus long du répertoire en secondes.
        """

        if not directory:
            directory = self.dir_input

        for pokemon in os.listdir(directory):
            pokemon_dir = os.path.join(directory, pokemon)

            if not os.path.isdir(pokemon_dir):
                continue

            for cri in os.listdir(pokemon_dir):
                chemin_cri = os.path.join(pokemon_dir, cri)

                if os.path.isfile(chemin_cri) and ".mp3" in cri:
                    duree, _, _ = self.get_duree_audio(chemin_cri)
                    self.durees.append(duree)

        duree_max = max(self.durees)

        return duree_max

    def pad_mono_audio(
        self,
        chemin_audio,
        duree_cible,
    ):
        """
        Prend un audio et effectue un padding pour que l'audio atteigne une durée qu'on a spécifié.
        Si l'audio est en stéréo le passe en mono.
        L'audio peut éventuellement etre reechantillonne selon le sr_fixe qu'on a défini dans l'init de la classe.
        Le padding que j'ai mit centre le cri car j'ai vu des etudes empiriques qui disaient que ça améliorait la robustesse alors j'essaye plutot que de padder a la fin.

        Parametres :
        - chemin_audio (str) : chemin de l'audio dont on veut modifier la longueur.
        - duree_cible (float) : durée finale qu'aura notre audio.

        Output :
        - padded_wf (torch.Tensor) : waveform apres padding.
        - sr (int) : sample rate de l'audio.
        """

        duree, wf, sr = self.get_duree_audio(chemin_audio)

        if self.sr_fixe and sr != self.sr_fixe:
            resampler = torchaudio.transforms.Resample(
                orig_freq = sr, new_freq = self.sr_fixe
            )
            wf = resampler(wf)
            sr = self.sr_fixe

        if not duree_cible:
            duree_cible = self.get_duree_max()

        if wf.shape[0] > 1:
            wf = torch.mean(wf, dim = 0, keepdim = True)

        if duree != duree_cible:
            padding_total = int(duree_cible * sr - wf.shape[1])
            padding_left = padding_total // 2
            padding_right = padding_total - padding_left
            padded_wf = torch.nn.functional.pad(
                wf, (padding_left, padding_right), mode = "constant", value = 0
            )

        else:
            padded_wf = wf

        return padded_wf, sr

    def pad_all_audios(self):
        """
        Applique le padding à tous les audios du dir_input.

        Parametres :
        Aucun, on applique juste la transfo récursivement sur tous les dirs du dir_input.

        Output :
        Application de la transfo.
        """

        duree_max = self.get_duree_max()

        for pokemon in os.listdir(self.dir_input):
            pokemon_dir = os.path.join(self.dir_input, pokemon)

            if not os.path.isdir(pokemon_dir):
                continue

            for cri in os.listdir(pokemon_dir):
                if not ".mp3" in cri:
                    continue

                chemin_cri = os.path.join(pokemon_dir, cri)
                padded_wf, sr = self.pad_mono_audio(chemin_cri, duree_max)

                torchaudio.save(chemin_cri, padded_wf, sr)

    def split_dataset(self):
        """
        (Fonction que j'ai récupéré de mon projet sur le CNN du pokedex lol on recycle)
        Sépare un dataset d'audios en trois sous-dossiers : train, val et test.

        Arborescence initiale :
        pokemon/
        │
        ├── abra/
        │   ├── [...].mp3
        │   ├── [...].mp3
        ├── pikachu/
        │...

        Arborescence après séparation :
        pokemon_split/
        │
        ├── train/
        │   ├── abra/
                ├── [...].mp3
                ├── [...].mp3
        │   ├── pikachu/
        │...
        ├── val/
        │   ├── abra/
                ├── [...].mp3
        │   ├── pikachu/
        |...
        ├── test/
        │   ├── abra/
        │       ├── [...].mp3
        │   ├── pikachu/
        │...

        """

        os.makedirs(os.path.join(self.dir_output), exist_ok=True)

        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.dir_output, split), exist_ok=True)

        for class_dir in os.listdir(self.dir_input):
            class_path = os.path.join(self.dir_input, class_dir)

            if not os.path.isdir(class_path):
                continue

            files = np.array(
                [
                    f
                    for f in os.listdir(class_path)
                    if os.path.isfile(os.path.join(class_path, f))
                ]
            )
            np.random.shuffle(files)
            shuff_files = files.tolist()

            n_total = len(shuff_files)
            n_train = int(self.train_size * n_total)
            n_val = int(self.val_size * n_total)

            splits = {
                "train": shuff_files[:n_train],
                "val": shuff_files[n_train : n_train + n_val],
                "test": shuff_files[n_train + n_val :],
            }

            for split, split_files in splits.items():
                split_class_dir = os.path.join(self.dir_output, split, class_dir)
                os.makedirs(split_class_dir, exist_ok=True)
                for fname in split_files:
                    shutil.copy(
                        os.path.join(class_path, fname),
                        os.path.join(split_class_dir, fname),
                    )


# Utilisation pour notre cas
if __name__ == "__main__":
    np.random.seed(77)
    BASE_DIR = os.getenv("BASE_DIR")
    dir_input = os.path.join(BASE_DIR, "data/screams")
    dir_output = os.path.join(BASE_DIR, "data/split_data")
    train_size = 0.7
    val_size = 0.2
    sr_fixe = (
        16000  # Dans la doc de AST ils disent que le modele attend des sr de 16000
    )

    ap = AudioPreprocessing(
        dir_input=dir_input,
        dir_output=dir_output,
        train_size=train_size,
        val_size=val_size,
        sr_fixe=sr_fixe,
    )

    ap.pad_all_audios()
    ap.split_dataset()
