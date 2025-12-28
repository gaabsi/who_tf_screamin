import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import ASTFeatureExtractor

from modele import ModeleEntrainement
from preprocessing import AudioPreprocessing


class ModelEvalMetriques:
    """
    Classe qui permet de digérer les logs generees par lightning pendant l'entrainement.
    Resume l'entrainement et les performances (ici selon la loss et l'acu) entre le train et le val set.
    A la fin on sauvegarde le plot pour savoir si notre modele a potentiellement overfit, si il devait tourner plus lgtps, ...
    """

    def __init__(self, chemin_logs, chemin_output, nom_figs):
        self.chemin_logs = chemin_logs
        self.chemin_output = chemin_output
        self.nom_figs = nom_figs

    def process_logs(self):
        """
        Fonction de processing des logs de lightning pour avoir quelque chose de plus compact.
        Met tous les logs de notre entrainementdans un pd.Dataframe qui résume train et val accu et loss a chaque fin d'epoch.
        """

        df = pd.read_csv(self.chemin_logs)
        check_condition = ["train_acc_epoch", "val_acc"]
        cols = ["epoch", "train_acc_epoch", "train_loss_epoch", "val_acc", "val_loss"]

        df_logs = (
            df[df[check_condition].notna().any(axis=1)][cols]
            .groupby("epoch")
            .mean()
            .reset_index()
        )

        return df_logs

    def plot_logs(self):
        """
        Fonction pour plot les courbes de loss et d'acc sur les train et val dataset.
        La figure finale sera enregistrée au format png.
        """

        df_logs = self.process_logs()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(
            df_logs["epoch"],
            df_logs["train_loss_epoch"],
            "o-",
            label="Train Loss",
            linewidth=2,
        )
        ax1.plot(
            df_logs["epoch"], df_logs["val_loss"], "s-", label="Val Loss", linewidth=2
        )
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            df_logs["epoch"],
            df_logs["train_acc_epoch"],
            "o-",
            label="Train Acc",
            linewidth=2,
        )
        ax2.plot(
            df_logs["epoch"], df_logs["val_acc"], "s-", label="Val Acc", linewidth=2
        )
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.set_title("Training & Validation Acc", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            os.path.join(self.chemin_output, self.nom_figs), bbox_inches="tight"
        )
        plt.close(fig)


class PredictEval:
    """
    Cette classe permet d'effectuer une prediction sur de nouvelles donnees.
    On refait la pipeline dans le sens inverse.
    """

    def __init__(self, checkpoint_path, feature_extractor, csv_path):
        self.checkpoint_path = checkpoint_path
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(feature_extractor)
        self.csv_path = csv_path
        self.duree_audio_train = 2.772
        self.modele = None
        self.index_nom = None
        self.audio_preprocess = AudioPreprocessing(
            dir_input=None,
            dir_output=None,
            train_size=None,
            val_size=None,
            sr_fixe=16000,
        )

    def custom_modele(self):
        """
        Recupere la classe d'entrainement du modele et utilise la classe parente de lightning qui nous permet de load sur un checkpoint.
        Ca nous permet de mettre les poids de notre modele qu'on a custom en amont de cette etape.
        """
        if not self.modele:
            self.modele = ModeleEntrainement.load_from_checkpoint(self.checkpoint_path)
            self.modele.eval()

        return self.modele

    def map_index_nom(self):
        """
        On refait rapidement le mapping index : nom dans un dictionnaire
        On reutilise un bout de code qu'on avait dev dans le modele.py
        """
        if not self.index_nom:
            pokedex = pd.read_csv(
                self.csv_path,
                usecols=["INT_numéro", "Pokémon"],
                dtype={"INT_numéro": int, "Pokémon": str},
                encoding="utf-8",
            )
            self.index_nom = dict(zip(pokedex["INT_numéro"] - 1, pokedex["Pokémon"]))

        return self.index_nom

    def predict(self, audio_path):
        """
        Ici on fait une fonction qui nous permet de prendre un audio en input et de sortir la classe predite
        On repasse la fonction de preprocessing a notre input comme ça on peut prendre n'importe quel audio en input.
        """

        modele = self.custom_modele()
        index_nom = self.map_index_nom()

        wf, sr = self.audio_preprocess.pad_mono_audio(
            chemin_audio=audio_path, duree_cible=self.duree_audio_train
        )
        wf = wf.squeeze(0)

        inpoute = self.feature_extractor(
            wf.numpy(), sampling_rate=sr, return_tensors="pt"
        )

        impoute_model = inpoute["input_values"]

        with torch.no_grad():
            logits = modele(impoute_model)
            probas = torch.softmax(logits, dim=1)

            conf, pred_index = torch.max(probas, dim=1)

            pred_index = pred_index.item()
            conf = conf.item()

        nom_pred = index_nom[pred_index]
        vrai_nom = os.path.dirname(audio_path).split("/")[-1]
        correct = int(vrai_nom == nom_pred)

        return nom_pred, vrai_nom, conf, correct


if __name__ == "__main__":
    BASE_DIR = os.getenv("BASE_DIR")
    lighning_dir = os.path.join(BASE_DIR, "models/version_0/lightning_logs")
    chemin_logs = os.path.join(lighning_dir, "metrics.csv")
    test_dir = os.path.join(BASE_DIR, "data/split_data/test")
    checkpoint_path = os.path.join(BASE_DIR, "models/version_0/version_0.ckpt")
    feature_extractor = "MIT/ast-finetuned-audioset-10-10-0.4593"
    CSV_PATH = os.getenv("CSV_PATH")
    test_perf_csv_path = os.path.join(lighning_dir, "test_perf.csv")

    eval = ModelEvalMetriques(
        chemin_logs=chemin_logs, chemin_output=lighning_dir, nom_figs="eval_modele.png"
    )
    eval.plot_logs()

    test_set_eval = PredictEval(
        checkpoint_path=checkpoint_path,
        feature_extractor=feature_extractor,
        csv_path=CSV_PATH,
    )
    cris = [
        os.path.join(test_dir, pokemon, cri)
        for pokemon in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, pokemon))
        for cri in os.listdir(os.path.join(test_dir, pokemon))
        if ".mp3" in cri
    ]

    resultats = []
    for cri in cris:
        nom_pred, vrai_nom, conf, correct = test_set_eval.predict(cri)
        resultats.append(
            {
                "fichier": os.path.basename(cri),
                "vrai_nom": vrai_nom,
                "nom_predit": nom_pred,
                "conf": conf,
                "correct": correct,
            }
        )

    test_perf = pd.DataFrame(resultats)
    accu = test_perf["correct"].mean()
    test_perf.to_csv(test_perf_csv_path)

    print(f"Accu de {accu:.0%} sur le test set")
