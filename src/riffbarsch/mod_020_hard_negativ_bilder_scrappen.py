# image_downloader.py
# Lädt automatisch Bilder zu angegebenen Suchbegriffen herunter
# Benötigt: pip install icrawler

from icrawler.builtin import BingImageCrawler
import os
import requests
import urllib3
import time
import random

# Liste deiner Hard-Negative Suchbegriffe
search_terms = [

    # --- Land / Luft ---
    # 11) Wespen & Insekten
    "wasp closeup stripes",
    "hornet black yellow stripes",
    "bee striped abdomen macro",
    "yellow jacket wasp flying",
    "striped hoverfly insect",

    # 12) Mücken & andere Fluginsekten
    "mosquito closeup striped legs",
    "tiger mosquito stripes",
    "striped fly macro",
    "dragonfly striped wings",
    "gnat swarm macro",

    # 13) Zebras & gestreifte Säugetiere
    "zebra stripes closeup",
    "zebra herd savanna",
    "zebra skin pattern",
    "striped okapi animal",
    "striped hyena fur pattern",

    # 14) Zäune & Strukturen
    "striped fence wood",
    "picket fence black white",
    "striped gate metal",
    "lattice fence pattern",
    "striped wall pattern",

    # 15) Andere gestreifte Tiere
    "tiger stripes closeup",
    "striped cat fur",
    "chipmunk stripes",
    "skunk black white stripes",
    "badger stripes face",

    # 16) Vögel mit Streifenmustern
    "striped bird feathers",
    "cuckoo bird striped wings",
    "owl striped plumage",
    "finch striped chest",
    "hawk striped tail feathers",

    # 17) Gestreifte Aliens (Sci-Fi / Kunst)
    "striped alien creature concept art",
    "sci fi striped extraterrestrial",
    "alien insect stripes",
    "fantasy striped monster",
    "black white striped alien humanoid",
]

search_terms1 = [
    # 1) Andere gestreifte Fische
    "sergeant major fish reef",
    "Abudefduf saxatilis underwater",
    "Heniochus butterflyfish reef",
    "black and white butterflyfish underwater",
    "ribboned sweetlips fish",
    "sweetlips striped fish underwater",

    # 2) Jungfische mit gelben Körpern + dunklen Streifen
    "juvenile golden trevally fish",
    "juvenile trevally yellow stripes",
    "young golden trevally underwater",
    "yellow striped juvenile reef fish",

    # 3) Gestreifte Bodentexturen / Korallenmuster
    "striped coral reef pattern",
    "coral reef textures underwater",
    "algae striped pattern underwater",
    "reef rock banded pattern",
    "sea sponge striped texture",

    # 4) Andere Meerestiere mit Streifen oder kontrastierenden Mustern
    "banded sea snake underwater",
    "zebra moray eel underwater",
    "striped sea cucumber",
    "striped sea urchin spines",
    "anemone tentacles striped pattern",

    # 5) Menschliche Gegenstände mit Streifen / gelben Flächen
    "diver striped wetsuit underwater",
    "scuba diver yellow fins",
    "snorkeler yellow mask",
    "yellow buoy floating water",
    "striped towel beach",

    # 6) Schwärme ähnlicher Fische
    "sergeant major fish school",
    "reef fish striped school underwater",
    "shoal of striped tropical fish",
    "Abudefduf fish group reef",
    "sweetlips fish swarm underwater",

    # 7) Man-made Patterns
    "wooden fence with stripes",
    "striped beach towel",
    "striped fabric pattern",
    "sunshade striped pattern",
    "warning sign yellow black stripes",

    # 8) Beleuchtung / Color-shift Negatives
    "underwater backscatter photography",
    "reef underwater shadow contrast",
    "underwater low light photography",
    "color shifted underwater coral",
    "greenish underwater reef scene",

    # 9) Teilverdeckte / Fragmentierte Fische
    "reef fish closeup head",
    "fish tail closeup underwater",
    "fish dorsal fin closeup underwater",
    "cropped fish underwater photo",
    "partial fish body underwater",

    # 10) 3D-Objekte und Aquarium-Dekoration
    "plastic aquarium fish decoration",
    "artificial reef fish decor",
    "aquarium ornament striped fish",
    "fake fish toy underwater",
    "aquarium background plants plastic",
]


# Gesamtzahl pro Suchbegriff
num_images = 50  

for term in search_terms:
    print(f"Suche nach '{term}' ...")
    crawler = BingImageCrawler(storage={"root_dir": f"downloads/{term.replace(' ', '_')}"})
    crawler.crawl(keyword=term, max_num=num_images)

zielordner = r"E:\dev\projekt_python_venv\010_Riffbarsch\datasets\raw\i_net"
os.makedirs(zielordner, exist_ok=True)

# Warnungen zu unsicherem SSL ignorieren (nur für Testzwecke!)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_bild(url, zielpfad):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, stream=True, headers=headers, timeout=15, verify=False)
        if response.status_code == 200:
            with open(zielpfad, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Heruntergeladen: {zielpfad}")
        else:
            print(f"Fehler {response.status_code} beim Download: {url}")
    except Exception as e:
        print(f"Fehler bei {url}: {e}")

# Beispiel-URLs (ersetzen durch echte Bild-Links)
bild_urls = [
    "https://inaturalist-open-data.s3.amazonaws.com/photos/117360580/original.jpg",
    "https://i.pinimg.com/originals/6c/77/5f/6c775fd459df007f0b3a5262096a008b.jpg",
    "https://www.ark.au/images/species/fish/black-and-white-butterflyfish-1.jpg"
]

for url in bild_urls:
    dateiname = os.path.basename(url.split('?')[0])
    zielpfad = os.path.join(zielordner, dateiname)
    download_bild(url, zielpfad)
    # Pause zwischen 2 und 5 Sekunden, um menschliches Verhalten zu simulieren
    time.sleep(random.uniform(2, 5))
