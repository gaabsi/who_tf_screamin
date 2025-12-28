import os
import warnings

warnings.simplefilter("ignore")

import pandas as pd
import requests
from bs4 import BeautifulSoup

POKEMON_VERSION = os.getenv("POKEMON_VERSION")
CSV_PATH = os.path.expanduser(os.getenv("CSV_PATH"))
SCREAMS_PATH = os.path.expanduser(os.getenv("SCREAMS_PATH"))
os.makedirs(SCREAMS_PATH, exist_ok = True)

# Récupérer le html de la page du pokedex de la gen1 (on peut mettre une autre gen ça marche aussi en mettant "legendes-za" par exemple)
url_page_pokemon = f"https://www.pokebip.com/pokedex/{POKEMON_VERSION}"
html_brut = requests.get(url_page_pokemon).text
soup = BeautifulSoup(html_brut, "html.parser")

lignes = soup.find_all("tr")[
    1:
]  # On prend toutes les lignes de notre tableau sauf le header
pokemons_data = []

for ligne in lignes:
    cells = ligne.find_all("td")

    name_tag = cells[1].find("a")
    nom = name_tag.text.strip()
    url = name_tag["href"]

    type_tags = cells[2].find_all("img")
    types = "/".join([t["alt"].strip() for t in type_tags])

    numero_natio = (
        cells[3].find("span", class_="has-tooltip").text.strip().split("#")[1]
    )

    pv = cells[4].text.strip()
    att = cells[5].text.strip()
    defense = cells[6].text.strip()
    special = cells[7].text.strip()
    vit = cells[8].text.strip()

    pokemons_data.append(
        {
            "Numero_pokédex_natio": numero_natio,
            "INT_numéro": int(numero_natio),
            "Pokémon": nom,
            "URL_page": url,
            "Type(s)": types,
            "PV": pv,
            "ATT": att,
            "DEF": defense,
            "SPEC": special,
            "VIT": vit,
        }
    )

pokemons_df = pd.DataFrame(pokemons_data)
pokemons_df.to_csv(CSV_PATH, index=False, encoding="utf-8")

# Pour récupérér les cris on va commencer par créer un dossier par pokémon, dans chaque dossier on stockera son cri pour la suite
for pokemon in pokemons_df["Pokémon"].unique():
    pokemon_dir = os.path.join(SCREAMS_PATH, pokemon)
    os.makedirs(pokemon_dir, exist_ok=True)

    numero_natio = pokemons_df[pokemons_df["Pokémon"] == pokemon]["INT_numéro"].iloc[0]
    url = f"https://pokebip.com/audio/cris-sl/{numero_natio}.mp3" # C'est ici que le cri est stocké on peut le voir en inspectant l'html

    resp = requests.get(url)
    resp.raise_for_status()
    with open(os.path.join(pokemon_dir, f"{pokemon}.mp3"), "wb") as cri:
        cri.write(resp.content)
