import os
import warnings

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

warnings.filterwarnings("ignore", category=UserWarning)
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


class AudioAugmentation:
    """
    Principe :
    - On part d'un fichier `dir_cris/<pokemon>/<pokemon>.mp3` qu'on a scrapé.
    - On applique entre 1 et N transformations aléatoires (stretch, pitch shift, filtres, bruit)
    - On sauvegarde les nouveaux fichiers dans `dir_cris/<pokemon>/` jusqu'à atteindre `len_dir_cible`.

    Parametres :
    - augmentation_rules : (dict)
        Dictionnaire de règles, par ex :
        - "stretch": [min_ratio, max_ratio]
        - "p_shift": array/list de demi-tons possibles
        - "HighPass": [min_cutoff, max_cutoff]
        - "LowPass": [min_cutoff, max_cutoff]
        - "snr": [min_snr, max_snr]
    - chemins_bruits : (list)
        Liste de chemins vers des fichiers de bruit/musique utilisés pour `add_custom_noise`.
    - dir_cris : (str)
        Répertoire racine des cris (contient un sous-dossier par Pokémon).
    - pokemon : (str)
        Nom du Pokémon (sert à trouver le fichier source et à nommer les sorties).
    - len_dir_cible : (int)
        Nombre total de fichiers visés dans `dir_cris/<pokemon>/` (inclut le fichier source).
    """

    def __init__(
        self, augmentation_rules, chemins_bruits, dir_cris, pokemon, len_dir_cible
    ):

        self.augmentation_rules = augmentation_rules
        self.chemins_bruits = chemins_bruits
        self.dir_cris = dir_cris
        self.pokemon = pokemon
        self.len_dir_cible = len_dir_cible
        self.log = []

    def _reset_log(self):

        self.log = []

    def load_audio(self, chemin_audio):
        """
        Prend un fichier son en input et nous renvoie la waveform et le sample rate associé pour retravailler l'audio.

        Parametres :
        - chemin_audio (str) : chemin de l'audio qu'on veut load.

        Output :
        - waveform (torch.Tensor) : amplitude du son dans le temps
        - sample_rate (int) : fréquence qui renseigne sur le nombre d'échantillons par seconde dans ce signal
        """

        waveform, sample_rate = torchaudio.load(chemin_audio, backend="soundfile")

        return waveform, sample_rate

    def stretch(self, wf, sr):
        """
        Permet de modifier la vitesse de lecture d'un audio.
        On peut ainsi jouer le meme audio plus ou moins vite selon un facteur tiré aléatoirement entre 2 bornes.
        Les bornes permettent d'éviter d'avoir un audio lu en x20 par exemple et d'avoir un audio completement dénaturé.

        Parametres :
        - wf (torch.Tensor) : le waveform de l'audio
        - sr (int) : le sample rate de l'audio

        Output :
        - waveform_stretch (torch.Tensor) : le waveform une fois étiré ou compressé
        - sr (int) : sample rate de l'audio
        """

        stretch_rules = self.augmentation_rules["stretch"]
        stretch_ratio = np.random.uniform(stretch_rules[0], stretch_rules[1])
        spectrogram = T.Spectrogram(power=None)
        stretch = T.TimeStretch()
        inverse = T.InverseSpectrogram()

        spectro = spectrogram(wf)
        stretched = stretch(spectro, overriding_rate=1 + stretch_ratio)
        waveform_stretch = inverse(stretched)

        self.log.append(f"x{1+stretch_ratio:.2f}")
        return waveform_stretch, sr

    def pitch_shift(self, wf, sr):
        """
        Permet de monter/descendre globalement le ton de l'audio.
        Si on le monte de n demi-tons on aura un audio globalement plus aigu.
        Inversement, si on le baisse de n demi-tons il sera plus grave.
        Dans notre cas le n est aléatoire, positif ou négatif et tiré entre 2 bornes pour ne pas dénaturer l'audio.

        Parametres :
        - wf (torch.Tensor) : waveform de l'audio
        - sr (int) : sample rate de l'audio

        Output :
        - wf_shifted (torch.Tensor) : waveform de l'audio une fois le pitch shift de n demi-tons appliqué
        - sr (int) : le sample rate de l'audio
        """

        ps_rules = self.augmentation_rules["p_shift"]
        nb_tons = np.random.choice(ps_rules)

        p_shift = T.PitchShift(sample_rate=sr, n_steps=nb_tons)
        wf_shift = p_shift(wf)

        self.log.append(f"shift_{nb_tons}")
        return wf_shift, sr

    def hilo_pass_filter(self, wf, sr):
        """
        Permet d'appliquer un filtre High ou Low Pass :
        - le HighPass atténue les fréquences en dessous d'un seuil, tout ce qui est plus haut passe.
        - inversement pour le LowPass qui laisse passer sans modifier toutes les fréquences en dessous du seuil.
        Le type de filtre est tiré aléatoirement.
        La fréquence de masquage est également tirée aléatoirement parmi des plages de fréquences produisant des résultats crédibles.
        On crée seulement des variantes un peu plus étouffées ou stridentes mais toujours réalistes.

        Parametres :
        - wf (torch.Tensor) : waveform de l'audio
        - sr (int) : sample rate de l'audio

        Output :
        - wf (torch.Tensor) : waveform de l'audio avec le filtre (High/Low) appliqué
        - sr (int) : sample rate de l'audio
        """

        filter_type = np.random.choice(["HighPass", "LowPass"])
        plage = self.augmentation_rules[filter_type]
        cutoff = np.random.randint(plage[0], plage[1])

        if filter_type == "HighPass":
            wf = F.highpass_biquad(wf, sr, cutoff_freq=cutoff)

        else:
            wf = F.lowpass_biquad(wf, sr, cutoff_freq=cutoff)

        self.log.append(f"{filter_type}_{cutoff}")
        return wf, sr

    def add_custom_noise(self, wf, sr):
        """
        Ajoute un bruit de fond a notre audio parmi une bibliotheque de bruit spécifiée.
        Le bruit en question est tiré aléatoirement parmi la bibliotheque de bruits.
        Le ratio son : bruit (tiré aléatoirement) est plutot élevé par défaut ce qui permet au son de toujours avoir le dessus sur le bruit.
        Si le bruit est plus long que l'audio qu'on veut lui mixer on découpe une partie aléatoire du bruit qu'on passera en fond.
        Toutes ces parties aléatoires permettent une très grande quantité de combinaisons possibles rien qu'avec ce bruit.

        Parametres :
        - wf (torch.Tensor) : waveform de l'audio
        - sr (int) : sample rate de l'audio

        Output :
        - wf_mix (torch.Tensor) : waveform "mixé" de l'audio avec le bruit de fond appliqué
        - sr (int) : sample rate de l'audio
        """

        bruit = np.random.choice(self.chemins_bruits)
        plage_snr = self.augmentation_rules["snr"]
        ratio_son_bruit = torch.tensor([np.random.randint(plage_snr[0], plage_snr[1])])
        snr_ratio = int(ratio_son_bruit.item())
        nom_bruit = os.path.splitext(os.path.basename(bruit))[0]

        wf_bruit, sr_bruit = self.load_audio(bruit)

        if wf_bruit.shape[1] != wf.shape[1]:
            debut = np.random.randint(wf_bruit.shape[1] - wf.shape[1])
            fin = debut + wf.shape[1]
            new_wf = wf_bruit[:, debut:fin].mean(dim=0, keepdim=True)

        else:
            new_wf = wf_bruit.mean(dim=0, keepdim=True)
            debut, fin = 0, wf.shape[1]

        wf_mix = F.add_noise(wf, new_wf, snr=ratio_son_bruit)
        self.log.append(f"noise_{snr_ratio}*{nom_bruit}_{debut}_{fin}")
        return wf_mix, sr

    def pioch_augmentation(self):
        """
        L'objectif ici est de tirer un nombre k (avec k>1) d'augmentations parmi nos n types d'augmentations implémentées précédemment.
        Un meme audio peut donc etre modifié entre 1 et 4 fois pour notre cas.
        Toutes les parties aléatoires de nos methodes sont conservées, on effectue ici un simple tirage sur lesquelles seront appliquées.

        Output :
        - wf (torch.Tensor) : on récupère notre waveform avec toutes les transformations appliquées.
        - sr (int) : encore une fois, on récupere le sample rate de notre audio
        - on renvoie une sorte d'id qui nous servira a comprendre quelles transformations ont été appliquées a chaque variante de notre audio
        """

        self._reset_log()
        wf, sr = self.load_audio(
            os.path.join(self.dir_cris, self.pokemon, self.pokemon + ".mp3")
        )

        transfo_possibles = [
            self.stretch,
            self.pitch_shift,
            self.hilo_pass_filter,
            self.add_custom_noise,
        ]
        n = np.random.randint(1, len(transfo_possibles) + 1)
        transfo_choisies = np.random.choice(
            len(transfo_possibles), size=n, replace=False
        )

        for transfo in transfo_choisies:
            wf, sr = transfo_possibles[transfo](wf, sr)

        return wf, sr, "_".join(self.log)

    def save_audio(self, wf, sr, tag):
        """
        Ici on prend notre audio sous forme de waveform et sample rate, on regarde via les logs quelles transfo lui ont été appliquées et on le sauvegarde.
        Le nom que prendra l'audio rendra compte des transformations qu'il a subi.

        Parametres :
        - wf (torch.Tensor) : waveform de l'audio a enregistrer.
        - sr (int) : sample rate de l'audio a enregistrer
        - tag (str) : les logs des transfo qu'on a appliquées, qui serviront a nommer l'audio qu'on enregistre.

        Output :
        Ne retourne rien, ecrit simplement l'audio.
        """

        filename = f"{self.pokemon}_{tag}.mp3"
        path = os.path.join(self.dir_cris, self.pokemon, filename)
        torchaudio.save(path, wf, sr)
        return path

    def fill_dir(self):
        """
        Méthode qui orchestre en quelque sorte le "spam" de notre processus d'augmentation jusqu'a arriver au nombre d'audio qu'on voulait.
        Exemple :
        On a un cri de Ratatta, j'en veux 5 au total en comptant le cri original
        Cette méthode nous permettra de créer les 4 autres cris de manière artificielle et s'arrêtera quand nous aurons nos 5 audios.

        Update sur un point d'attention :
        En bouclant sur les modifs aléatoires il m'est arrivé de produire des waveform qui avaient des NaN ou des Inf donc impossible à save.
        En enchainant les transfo ça a du créer une mutation de fou de l'audio donc j'ai rajouté un check avant la save pour pallier ce probleme.
        On aurait pu forcer en sauvegardant en .wav (on aurait pas lever d'erreur lors de l'écriture)
        mais premierement c'est beaucoup plus lourd et en plus on aurait eu un audio corrompu dedans.
        Donc pas foufou pour entrainer un reconnaisseur de cri si on lui donne un audio corrompu en entrée et qu'on le labellise comme un Rattata par exemple.
        """

        dir_cri_poke = os.path.join(self.dir_cris, self.pokemon)
        nb_cri = len(os.listdir(dir_cri_poke))

        with torch.no_grad():
            while nb_cri < self.len_dir_cible:
                wf, sr, tag = self.pioch_augmentation()
                if (
                    isinstance(wf, torch.Tensor)
                    and (not wf.is_complex())
                    and bool(torch.isfinite(wf).all().item())
                ):
                    filename = f"{self.pokemon}_{tag}.mp3"
                    fichier = os.path.join(self.dir_cris, self.pokemon, filename)

                    if os.path.exists(fichier):
                        continue
                    else:
                        self.save_audio(wf, sr, tag)
                        nb_cri += 1
                else:
                    continue


# On peut prendre un petit exemple avec Rattata (mon pokemon préféré)

""" 
if __name__ == "__main__":

    
    # Donc la on spécifie tout ce qu'il nous faut pour initialiser la classe
    dir_des_bruits = os.path.expanduser("~/who_tf_screamin/data/battle_themes")
    chemins_bruits = [
        os.path.join(dir_des_bruits, musique)
        for musique in os.listdir(dir_des_bruits)
        if ".mp3" in musique
    ]
    regles = {
        "stretch": [-0.15, 0.15],
        "p_shift": np.arange(-3, 4, 1),
        "HighPass": [300, 1500],
        "LowPass": [3000, 6000],
        "snr": [15, 20],
    }
    dir_cri = os.path.expanduser("~/who_tf_screamin/data/screams")
    pokemon = "Rattata"
    len_dir_cible = 10

    # Init de la classer
    aud = AudioAugmentation(
        augmentation_rules=regles,
        chemins_bruits=chemins_bruits,
        dir_cris=dir_cri,
        pokemon=pokemon,
        len_dir_cible=len_dir_cible,
    )

    # Appel pour creer toutes nos variations de cri
    aud.fill_dir()
"""

# Apres pour boucler on lance un
# for pokemon in [poke for poke in os.listdir(dir_cri) if os.path.isdir(os.path.join(dir_cri, poke))] :

# Mais ça reste séquentiel pour faire toutes nos augmentations sur un pokemon -> pokemon suivant -> augmentations -> pokemon suivant ...
# donc si on veut faire bcp de pokemons ou si on veut faire bcp d'audios par pokemon ça risque d'etre très long
# On peut faire mieux : on parallelise la tache et pour eviter d'avoir des conflits etc on fait un worker par pokemon ça evite que 2 workers fassent la meme augmentation

# Donc on se fait une fonction qu'on pourra donner a nos workers


def fonction_worker(
    pokemon, dir_cri, chemins_bruits, len_dir_cible, augmentation_rules
):
    """
    Cette fonction nous permettra de concentrer un worker sur l'augmentation d'un pokemon.
    En faisant ça on evite que plusieurs workers travaillent sur le meme pokemon et créent les memes audios qui s'écrasent.

    Parametres :
    - pokemon (str) : le nom du pokemon dont le worker va s'occuper
    - dir_cri (str) : le dossier dans lequel le worker trouvera le cri du pokemon
    - chemin_bruits (liste) : liste des chemins des bruits (les musiques de fond) qu'on rajoutera peut etre à nos cris
    - len_dir_cible (int) : nombre d'échantillon de cri différents qu'on veut une fois l'augmentation finie
    - augmentation_rules (dict) : regles pour notre augmentation audio (type de modif : regles)

    Output :
    Pas vraiment d'output a proprement parler mais on retourne le nom du pokemon pour qu'on puisse voir s'il y en a un qui a planté.
    """

    seed = sum(ord(c) for c in pokemon)
    np.random.seed(seed)

    aud = AudioAugmentation(
        augmentation_rules=augmentation_rules,
        chemins_bruits=chemins_bruits,
        dir_cris=dir_cri,
        pokemon=pokemon,
        len_dir_cible=len_dir_cible,
    )

    aud.fill_dir()
    return pokemon


# Reel appel :

if __name__ == "__main__":

    BASE_DIR = os.getenv("BASE_DIR")
    dir_cri = os.path.join(BASE_DIR, "data/screams")
    dir_des_bruits = os.path.join(BASE_DIR, "data/battle_themes")

    len_dir_cible = 25

    regles = {
        "stretch": [-0.15, 0.15],
        "p_shift": np.arange(-3, 4, 1),
        "HighPass": [300, 1500],
        "LowPass": [3000, 6000],
        "snr": [15, 20],
    } # Pour comprendre ces plages tout est détaillé dans le notebook

    chemins_bruits = [
        os.path.join(dir_des_bruits, f)
        for f in os.listdir(dir_des_bruits)
        if f.lower().endswith(".mp3")
    ]

    pokemons = [
        p for p in os.listdir(dir_cri) if os.path.isdir(os.path.join(dir_cri, p))
    ]

    max_workers = min(
        4, os.cpu_count() - 1
    )  # Comme ça on peut faire autre chose a coté

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                fonction_worker,
                pokemon,
                dir_cri,
                chemins_bruits,
                len_dir_cible,
                regles,
            )
            for pokemon in pokemons
        ]

        for fut in as_completed(futures):
            pokemon = fut.result()
            print(f"{pokemon} tout est bon (*imagine un pouce en l'air*)", flush=True)
