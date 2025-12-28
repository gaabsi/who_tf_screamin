import json
import os
import subprocess

BASE_DIR = os.getenv("BASE_DIR")
url_playliste = (
    "https://www.youtube.com/playlist?list=PL2uxd6YWj7PLbKlHSlTO7Lv4E6csDZhRc"
)
theme_musique = "Battle"

playliste = subprocess.run(
    ["yt-dlp", "--flat-playlist", "-J", url_playliste],
    stdout=subprocess.PIPE,
    text=True,
    encoding="utf-8",
    check=True,
)

data = json.loads(playliste.stdout)
os.makedirs(os.path.join(BASE_DIR, "data/battle_themes"), exist_ok=True)

for video in data.get("entries", []):
    title = video.get("title")
    if theme_musique in title:
        url = video.get("url")
        clean_titre = title.split("-")[0].strip().replace(" ", "_")

        subprocess.run(
            [
                "yt-dlp",
                "--force-overwrites",
                "--no-continue",
                "-x",
                "--audio-format",
                "mp3",
                "-o",
                os.path.join(BASE_DIR, "data/battle_themes/", f"{clean_titre}.mp3"),
                url,
            ],
            check=True,
        )
