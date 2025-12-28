# who_tf_screamin

## Description 
Le but du projet est de faire un classifieur qui permet de reconnaitre un pokémon par son cri.  
Le modèle utilisé pour la classification est un *AST (Audio Spectrogram Transformer)*.  
Pour éviter d'avoir a réentrainer un modele from scratch j'ai récupéré un modele deja pré entrainé et seulement les dernieres couches servant a la classification on été dé-freeze pour l'adapter à notre tâche de classification.  

## Demonstration
Démonstration sur le cri d'aquali (vous pouvez verifier il est dans le test set, l'original a jamais été vu par le modele)

<p align="center">
  <a href="https://youtu.be/LO-0n_fZ2A8">
    <img src="https://img.youtube.com/vi/LO-0n_fZ2A8/0.jpg" alt="Démo Who TF Screamin" width="600">
  </a>
</p>

## Pipeline résumée

### Webscraping 
La premiere partie du projet consiste a récuperer les données dont nous aurons besoin.  
Il n'existait pas de dataset de cris de pokémon déjà pret sur kaggle (du moins pas au moment où j'ai eu l'idée de ce projet) il a donc fallut construire cette base.  
Pour ce faire, un site nommé pokebip recense toutes les données de tous les pokémons génération par génération (images, stats, cris, moovepool, ...)  
La structure du site étant très simple il a été très facile d'en extraire les données interessantes et d'obtenir ce qu'il nous fallait.  
A l'issue de cette phase on a donc : 
- un pokédex custom sur la génération de notre choix (la premiere pour nous) qui contient toutes les info (stats, numéro dans le dex, url de la page du pokémon, ...)
- un directory qui contient 151 pokémons (le nombre de pokemons dans le pokedex la 1G) et dans chaque directory se trouve le cri du pokemon au format .mp3.  

Donc ici on a toutes nos données initiales mais avec un seul échantillon par pokémon c'est impossible d'entrainer un classifieur donc on en a pas encore fini avec ces cris.  
La solution puisqu'on peut pas récuperer enormement d'echantillons de cri de pokemons ça va etre de les fabriquer nous meme en créant artificiellement des variantes qui pourraient vraiment exister d'un meme cri, on détaillera ça plus tard.

![scraping image](/img/scraping.png)

### Battle Theme Scraping 
Ça va commencer a spoil la partie suivante mais pour créer des echantillons réalistes des cris de pokemons, on peut ajouter un bruit de fond a notre cri pour simuler la musique d'ambiance qui se lance lors d'un combat pokémon.  
Ça nous permet a la fois de faire des cris qui sont réalistes car ces variantes peuvent vraiment se produire et d'avoir plus d'échantillons.  
Donc pour ce faire, j'ai trouvé une playlist youtube qui recensait toutes les musiques de la premiere generation et j'ai téléchargé les musiques de combat pokémon uniquement pour qu'on puisse éventuellement les ajouter a notre augmentation audio future.  

![battle theme scraping image](/img/battle_theme_scraping.png)

### Audio Augmentation
Ça fait déjà 2 parties qu'on le tease, ça y est on en parle !  
Donc l'objectif est de passer de 1 cri a au moins une vingtaine par pokémon pour que notre classifieur puisse apprendre a reconnaitre ces cris.  
Alors comment on fait pour créer artificiellement des audios ? Toutes les méthodes ont été expliquées dans le notebook d'exploration : 
- On joue sur la durée : un cri qui se joue plus ou moins vite (le facteur a été choisi pour ne pas dénaturer le cri et produire un cri plausible)
- On joue sur le pitch : on fait un cri plus aigu ou plus grave de quelques demi-tons (encore une fois de manière modérée)
- On passe un filtre High ou Low Pass : on rend le cri plus étouffé ou plus strident (toujours modérément)
- On ajoute un bruit parasite : on découpe une partie aléatoire de la musique de bataille et on l'ajoute au cri de base en jouant sur le rapport entre les deux. 

Une fois qu'on a défini toutes nos augmentations probables on prend un cri on l'envoie dans la boite magique, il tire aléatoirement 1 ou 4 augmentations qui sont elles memes aléatoires.  
On itère jusqu'a avoir autant de cris qu'on veut par pokémon et dans notre cas on a pris 25 cris par pokémon.  

![audio augmentation image](/img/audio_augmentation.png)

### Preprocessing
Une fois qu'on a tous nos cris, le but est d'uniformiser tous ces cris.  
- On prend donc l'audio le plus long qu'on a et on l'utilise comme référence, tous les audios qui sont plus courts que celui la on rajoutera des 0 au debut et a la fin (on garde le cri au milieu).
- Si notre audio est en stéréo on le passe en mono.
- On resample nos audios pour passer à 16.000 Hz (quasi aucun changement) mais on le fait nous meme plutot que de le laisser au modèle qui lui aurait fait un carnage à nos audios.  

Donc on applique toutes ces transformations à tous nos audios pour les standardiser.  
Une fois qu'on a fait cela on découpe nos données en 3 set (train, val et test) on le fait nous meme encore une fois pour pouvoir gérer finement ce processus et on se retrouve avec 3 dossiers qui contiennent nos audios scrapés, pré-processés et maintenant splités.  
Pour les tailles de ces différents split on est à 70% de train, 20% de validation et 10% de test.  

![preprocessing image](/img/preprocessing.png)

### Modelisation
Pour la partie modélisation, on a pris un type de modele qui marche super bien pour la classification audio : un AST (Audio Spectrogram Transformer), pour ne pas avoir a en entrainer un from scratch et capitaliser sur le travail qui a déjà été effectué on a réutilisé un AST fait par le MIT dont on a gardé le backbone (tout le modele sauf la tete de classification) et on a juste réentrainé la tete qui sert a predire la classe en sortie de modele.  
On garde donc les benefices de l'entrainement lourd du modèle (reconnaissance de patterns, ... ) et on le spécialise a notre tache de classification particuliere (transfer learning). 

![modelisation image](/img/modelisation.png)

### Evaluation
Pour évaluer notre modele on commence par regarder si nos performances sur le jeu de val suivent la meme trajectoire que celles sur le jeu de train.  
C'est notre cas, la loss et l'accuracy sont au coude a coude sur les 2 jeux donc notre modele n'a pas sur-appris.  
On voit que le gain marginal d'une epoch devient de plus en plus négligeable a partir de la 7-8e epoch mais ne stagne pas et ne déclanche pas non plus notre earlystopping donc on aurait meme pu continuer encore un peu le training si on voulait.  
Globalement notre modele semble avoir d'excellentes performances alors qu'on le rappelle on a au total que 25 audios par classe (dont 3 dont le modele n'a encore pas connaissance).  
On passe donc a la phase de prédiction sur les nouvelles données (celles de test) et on obtient un score de 96% de précision sur ce jeu de données ce qui colle avec les performances qu'on pouvait voir sur les courbes mentionnées précédemment.  
Globalement on a un modele très robuste et qui a d'excellentes perfs (gg à moi).  

![evaluation image](/img/evaluation.png)

## Structure du projet 
```
who_tf_screamin/
    ├── README.md 
    ├── explo.ipynb                     # Notebook d'exploration dans lequel je définis les regles d'augmentation et prend en main torchaudio
    ├── src/                            # Codes du projet
    │   ├── scraping.py                 # WebScraping du pokedex et de tous les cris des pokémons de la génération cible
    │   ├── music_theme_scraping.py     # Scraping des musiques de combat sur youtube
    │   ├── scream_even_more.py         # Augmentation audio (génération artificielle de cris probables) de 1 à 25 cris par pokémon
    │   ├── preprocessing.py            # Preprocessing audio (standardisation, split, ...)
    │   ├── modele.py                   # Modelisation et entrainement
    │   ├── evaluation.py               # Evaluation des performances du modele
    │   └── app.py                      # Streamlit ultra basique pour me la peter dans le README
    ├── models/                         # Stockage des modeles entrainés
    │   └── version_0/                  # Premiere (et unique) version du modele
    │       ├── version_O.ckpt          # Poids de notre premier modele (non plus mais sur Hugging Face Hub on le récupère juste après)
    │       └── lightning_logs/         # Logs de l'entrainement du modele
    │           ├── eval_modele.png     # Schémas pour monitorer l'eventuel l'over/under fit du modele
    │           ├── metrics.csv         # Logs générés par lightning
    │           └── test_perf.csv       # Prédictions détaillées (chaque pred sur le test set, sa proba, son true label, ...)
    ├── data/                           # Data dir (non push) : il est initialement scrapé puis successivement augmenté processé et splité 
    │   ├── split_data/                 # Dir qui contient les train, val et test data splités proprement 
    │   │   ├── train/                  # Audios utilisés pour le training
    │   │   ├── val/                    # Audios utilisés pour la validation
    │   │   └── test/                   # Audios utilisés pour le test
    │   ├── screams/                    # Cris de base, scrapés directement, sans aucune modification
    │   ├── battle_themes/              # Musiques de combats qu'on a scrapés
    │   └── pokedex.csv                 # Pokedex de la génération de pokémon qu'on a voulu scraper
    ├── img/                            # Schemas pour le coté waouw du README  
    │   ├── schema1.png
    │   └── ... 
    ├── Dockerfile                      # Dockerfile du projet
    ├── main.sh                         # Pour lancer tout le projet sans se casser la tete (entrainement gourmand, déconseillé en local, préferer une VM)
    ├── .gitignore                      # Fichiers a ignorer par Git 
    └── requirements.txt                # Packages du projet
```

## Clonage du projet 
Pour reproduire le projet et obtenir ses propres poids je conseille vivement de lancer ce dernier soit sur une machine avec un bon GPU soit d'en louer un (j'ai personnellement utilisé un pod de runpod avec une A40 qui coute 0.4$ de l'heure) que j'ai build a partir de mon image Docker dans Docker Hub, le temps d'entrainement du modele est d'environ 20mn.  
On peut s'y connecter via SSH, sortir les logs et les checkpoints soit via FTP soit en téléchargeant en clique bouton via l'UI de JupyterLab. 

Sinon pour juste faire tourner le modele en local en utilisant mes poids c'est beaucoup plus rapide et ça tourne sans probleme en local dans un conteneur.  

Pour ce deuxieme cas on peut le faire rapidement comme ça : 

```bash 
cd ~
git clone https://github.com/gaabsi/who_tf_screamin.git
```

Maintenant on récupere les poids du modèle qu'on a deja entrainé : 

```bash 
cd ~/who_tf_screamin
huggingface-cli download gaabsi/who_tf_screamin version_0.ckpt --local-dir ./models/version_0
rm -rf ./models/version_0/.cache/
```

On crée l'image Docker et on run : 

```bash 
cd ~/who_tf_screamin
docker buildx build --platform linux/amd64,linux/arm64 -t who_tf_screamin:latest . 
docker run --rm -v ~/who_tf_screamin:/app who_tf_screamin:latest  
```

Pour le premier cas (faire tourner le projet entier soi meme) il faudra modifier le main.sh et dé-commenter les lignes de training.  
Mais vraiment il faut un bon GPU pour le run en local. 