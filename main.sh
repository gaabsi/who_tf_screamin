#!/bin/bash 

# Pour que le script tourne aussi bien en local que dans son conteneur 
if [ -d "/app" ]; then
    BASE_DIR="/app"
    PYTHON="python"
else
    BASE_DIR="$HOME/who_tf_screamin"
    PYTHON="$BASE_DIR/venv/bin/python"
fi

# Variables qu'on utilisera pour la partie de récupération via scraping
POKEMON_VERSION="rouge-bleu"
CSV_PATH="$BASE_DIR/data/pokedex.csv"
SCREAMS_PATH="$BASE_DIR/data/screams"
PYTHON_DIR="$BASE_DIR/src"

# On exporte les variables dont on aura besoin dans nos scripts python
export BASE_DIR POKEMON_VERSION CSV_PATH SCREAMS_PATH

# Scraping de toutes les données dont on a besoin
$PYTHON "$PYTHON_DIR/scraping.py" # On récupere le pokedex complet de la génération et tous les cris (un dossier par pokemon)
$PYTHON "$PYTHON_DIR/music_theme_scraping.py" # Récupération de musiques du jeu sur youtube (playlist pour rouge bleu jaune)
 
# # Maintenant qu'on a tout ce qu'on veut on crée beaucoup de variantes de nos cris 
$PYTHON "$PYTHON_DIR/scream_even_more.py"

# Une fois qu'on a toutes nos données on les pre process : 
$PYTHON "$PYTHON_DIR/preprocessing.py"

# On entraine la derniere couche de notre modele AST sur nos audios
$PYTHON "$PYTHON_DIR/modele.py"

# On evalue les performances de notre modele
$PYTHON "$PYTHON_DIR/evaluation.py"